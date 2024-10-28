import os
import hashlib
import boto3
import sqlite3
import orjson
import argparse
import base64
import tempfile
import datetime
import posixpath
import threading
import logging
import boto3.session
import urllib3.exceptions

from dataclasses import dataclass
from pypdf import PdfReader
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from functools import partial
from typing import Optional, List, Tuple, Dict, Callable, Any
from urllib.parse import urlparse
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed

from pdelfin.data.renderpdf import render_pdf_to_base64png
from pdelfin.prompts import build_finetuning_prompt, PageResponse
from pdelfin.prompts.anchor import get_anchor_text
from pdelfin.s3_utils import parse_custom_id, expand_s3_glob, get_s3_bytes, parse_s3_path


# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global s3 client for the whole script, feel free to adjust params if you need it
workspace_s3 = boto3.client('s3')
pdf_s3 = boto3.client('s3')

# Quiet logs from pypdf and smart open
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("smart_open").setLevel(logging.ERROR)


class DatabaseManager:
    @dataclass(frozen=True)
    class BatchInferenceRecord:
        inference_s3_path: str
        pdf_s3_path: str
        page_num: int  # 1 indexed!
        round: int
        start_index: int
        length: int
        finish_reason: str
        error: Optional[str]

        def is_usable(self):
            return self.error is None and self.finish_reason == "stop"

    @dataclass(frozen=True)
    class PDFRecord:
        s3_path: str
        num_pages: int
        status: str

    def __init__(self, s3_workspace: str, skip_init: bool=False):
        cache_key = hashlib.sha256(s3_workspace.strip().lower().encode('utf-8')).hexdigest()
        home_cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'pdelfin', cache_key)
        os.makedirs(home_cache_dir, exist_ok=True)
        self.db_path = os.path.join(home_cache_dir, 'index.db')
        
        self.conn = sqlite3.connect(self.db_path)
        # Enable WAL mode so you can read and write concurrently
        self.cursor = self.conn.cursor()
        self.cursor.execute("PRAGMA journal_mode=WAL;")

        if not skip_init:
            self._initialize_tables()

    def _initialize_tables(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS page_results (
                inference_s3_path TEXT,
                pdf_s3_path TEXT,
                page_num INTEGER,
                round INTEGER,
                start_index BIGINT,
                length BIGINT,
                finish_reason TEXT,
                error TEXT
            )
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_path ON page_results(pdf_s3_path)
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_inf_path ON page_results(inference_s3_path)
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS pdfs (
                s3_path TEXT PRIMARY KEY,
                num_pages INTEGER,
                status TEXT DEFAULT 'pending'
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_files (
                s3_path TEXT PRIMARY KEY,
                etag TEXT
            )
        """)
        # Generic metadata such as current round
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        self.conn.commit()

    def get_metadata(self, key: str) -> Optional[str]:
        self.cursor.execute("SELECT value FROM metadata WHERE key=?", (key,))
        result = self.cursor.fetchone()
        return result[0] if result else None
    
    def set_metadata(self, key: str, value: str) -> None:
        self.cursor.execute("""
            INSERT INTO metadata (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """, (key, value))
        self.conn.commit()

    def is_file_processed(self, s3_path, etag):
        self.cursor.execute("SELECT etag FROM processed_files WHERE s3_path = ?", (s3_path,))
        result = self.cursor.fetchone()
        return result is not None and result[0] == etag
    
    def update_processed_file(self, s3_path, etag):
        self.cursor.execute("""
            INSERT INTO processed_files (s3_path, etag)
            VALUES (?, ?)
            ON CONFLICT(s3_path) DO UPDATE SET etag=excluded.etag
        """, (s3_path, etag))
        self.conn.commit()

    def clear_index(self):
        self.cursor.execute("""
            DELETE FROM processed_files;
        """)
        self.cursor.execute("""
            DELETE FROM page_results;
        """)
        self.conn.commit() 

    def add_index_entries(self, index_entries: List['BatchInferenceRecord']):
        if index_entries:
            self.cursor.executemany("""
                INSERT INTO page_results (inference_s3_path, pdf_s3_path, page_num, round, start_index, length, finish_reason, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [(entry.inference_s3_path, entry.pdf_s3_path, entry.page_num, entry.round, entry.start_index, entry.length, entry.finish_reason, entry.error) for entry in index_entries])
            self.conn.commit()

    def get_index_entries(self, pdf_s3_path: str) -> List['BatchInferenceRecord']:
        self.cursor.execute("""
            SELECT inference_s3_path, pdf_s3_path, page_num, round, start_index, length, finish_reason, error
            FROM page_results
            WHERE pdf_s3_path = ?
            ORDER BY inference_s3_path DESC, start_index ASC, page_num ASC
        """, (pdf_s3_path,))
        
        rows = self.cursor.fetchall()
        
        return [
            self.BatchInferenceRecord(
                inference_s3_path=row[0],
                pdf_s3_path=row[1],
                page_num=row[2],
                round=row[3],
                start_index=row[4],
                length=row[5],
                finish_reason=row[6],
                error=row[7]
            )
            for row in rows
        ]
    
    def delete_index_entries_by_inference_s3_path(self, inference_s3_path: str):
        self.cursor.execute("DELETE FROM page_results WHERE inference_s3_path = ?", (inference_s3_path,))
        self.conn.commit()

    def get_last_indexed_round(self) -> int:
        self.cursor.execute("""
            SELECT MAX(round)
            FROM page_results
        """)

        result = self.cursor.fetchone()
        return -1 if result[0] is None else result[0]

    def pdf_exists(self, s3_path: str) -> bool:
        self.cursor.execute("SELECT 1 FROM pdfs WHERE s3_path = ?", (s3_path,))
        return self.cursor.fetchone() is not None

    def add_pdf(self, s3_path: str, num_pages: int, status: str = 'pending') -> None:
        try:
            self.cursor.execute("""
                INSERT INTO pdfs (s3_path, num_pages, status)
                VALUES (?, ?, ?)
            """, (s3_path, num_pages, status))
            self.conn.commit()
        except sqlite3.IntegrityError:
            logger.warning(f"PDF with s3_path '{s3_path}' already exists.")

    def update_pdf_statuses(self, status_updates: Dict[str, str]) -> None:
        """
        Update the status of multiple PDFs in the database.

        :param status_updates: A dictionary where each key is an s3_path (str) and
                               each value is the new status (str) for that PDF.
        """
        self.cursor.executemany("""
            UPDATE pdfs
            SET status = ?
            WHERE s3_path = ?
        """, [(new_status, s3_path) for s3_path, new_status in status_updates.items()])
        self.conn.commit()

    def get_pdf(self, s3_path: str) -> Optional['PDFRecord']:
        self.cursor.execute("""
            SELECT s3_path, num_pages, status
            FROM pdfs
            WHERE s3_path = ?
        """, (s3_path,))
        
        row = self.cursor.fetchone()
        
        if row:
            return self.PDFRecord(
                s3_path=row[0],
                num_pages=row[1],
                status=row[2]
            )
        return None
    
    def get_pdfs_by_status(self, status: str) -> List['PDFRecord']:
        self.cursor.execute("""
            SELECT s3_path, num_pages, status
            FROM pdfs
            WHERE status == ?
            ORDER BY s3_path DESC, num_pages DESC
        """, (status, ))
        
        rows = self.cursor.fetchall()
        
        return [
            self.PDFRecord(
                s3_path=row[0],
                num_pages=row[1],
                status=row[2]
            )
            for row in rows
        ]

    def close(self):
        self.conn.close()


class BatchWriter:
    def __init__(
        self,
        output_prefix: str,
        max_size_mb: int = 250,
        after_flush: Optional[Callable[[List[Any]], Any]] = None,
    ):
        self.output_prefix = output_prefix
        self.max_size = max_size_mb * 1024 * 1024  # Convert MB to bytes
        self.batch_objects = []
        self.batch_size = 0
        self.after_flush = after_flush
        self.threads = []
        self.temp_file = None  # The temporary file object
        self.temp_file_path = None  # Path to the temporary file

        parsed = urlparse(output_prefix)
        self.is_s3 = parsed.scheme in ("s3", "s3a", "s3n")

        if not self.is_s3:
            os.makedirs(output_prefix, exist_ok=True)

    def write_line(self, obj: Optional[Any]):
        if obj is None:
            return

        line_bytes = orjson.dumps(obj)
        line_size = len(line_bytes) + 1  # +1 for newline

        if self.batch_size + line_size > self.max_size:
            self._write_batch()

        if self.batch_size == 0:
            # Open a new temporary file
            self.temp_file = tempfile.NamedTemporaryFile(mode="wb+", delete=False)
            self.temp_file_path = self.temp_file.name

        self.temp_file.write(line_bytes + b"\n")
        self.batch_objects.append(obj)
        self.batch_size += line_size

    def _write_batch(self):
        if self.batch_size == 0:
            return

        # Close the temp file
        self.temp_file.flush()
        self.temp_file.close()

        # Start a new thread to upload the temp file
        thread = threading.Thread(
            target=self._write_batch_to_file, args=(self.temp_file_path, self.batch_objects)
        )
        thread.start()
        self.threads.append(thread)

        # Reset batch_objects and batch_size
        self.batch_objects = []
        self.batch_size = 0
        self.temp_file = None
        self.temp_file_path = None

    def _write_batch_to_file(self, temp_file_path: str, batch_objects: List[Any]):
        # Compute hash based on file content
        hash_str = self._compute_hash(temp_file_path)
        output_path = self._get_output_path(hash_str)

        if self.is_s3:
            bucket, key = parse_s3_path(output_path)

            # Use the s3 client directly
            try:
                workspace_s3.upload_file(temp_file_path, bucket, key)
            except Exception as e:
                logger.error(f"Failed to upload {temp_file_path} to {output_path}: {e}", exc_info=True)
        else:
            # Move the temp file to the output path
            os.rename(temp_file_path, output_path)

        # After writing, call the after_flush callback if it is set
        if self.after_flush:
            self.after_flush(batch_objects)

        os.remove(temp_file_path)

    def _compute_hash(self, temp_file_path: str) -> str:
        """Compute a 20-character SHA1 hash of the file content."""
        sha1 = hashlib.sha1()
        with open(temp_file_path, "rb") as f:
            while True:
                data = f.read(1024*1024)
                if not data:
                    break
                sha1.update(data)
        return sha1.hexdigest()[:20]

    def _get_output_path(self, hash_str: str) -> str:
        """Generate the full output path with hash in the filename."""
        parsed = urlparse(self.output_prefix)
        if self.is_s3:
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            if key and not key.endswith("/"):
                key += "/"
            full_key = posixpath.join(key, f"output_{hash_str}.jsonl")
            return f"s3://{bucket}/{full_key}"
        else:
            filename = f"output_{hash_str}.jsonl"
            return os.path.join(self.output_prefix, filename)

    def close(self):
        self._write_batch()
        # Wait for all threads to finish
        for thread in self.threads:
            thread.join()


def build_page_query(local_pdf_path: str, pretty_pdf_path: str, page: int, target_longest_image_dim: int, target_anchor_text_len: int, image_rotation: int=0) -> dict:
    assert image_rotation in [0, 90, 180, 270], "Invalid image rotation provided in build_page_query"
    image_base64 = render_pdf_to_base64png(local_pdf_path, page, target_longest_image_dim=target_longest_image_dim)

    if image_rotation != 0:
        image_bytes = base64.b64decode(image_base64)
        with Image.open(BytesIO(image_bytes)) as img:
            rotated_img = img.rotate(-image_rotation, expand=True)

            # Save the rotated image to a bytes buffer
            buffered = BytesIO()
            rotated_img.save(buffered, format="PNG")

        # Encode the rotated image back to base64
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')


    anchor_text = get_anchor_text(local_pdf_path, page, pdf_engine="pdfreport", target_length=target_anchor_text_len)

    return {
        "custom_id": f"{pretty_pdf_path}-{page}",
        "chat_messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_finetuning_prompt(anchor_text)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ],
            }
        ],
    }


def process_jsonl_content(inference_s3_path: str) -> List[DatabaseManager.BatchInferenceRecord]:
    content_bytes = get_s3_bytes(workspace_s3, inference_s3_path)

    start_index = 0
    index_entries = []
    lines = content_bytes.splitlines(keepends=True)  # Split content into lines as bytes
    for line in lines:
        line_length = len(line)  # Length in bytes

        try:
            # Parse the line directly as JSON
            data = orjson.loads(line)
            pdf_s3_path, page_num = parse_custom_id(data["custom_id"])

            if data.get("completion_error", None) is not None:
                index_entries.append(DatabaseManager.BatchInferenceRecord(
                    inference_s3_path=inference_s3_path,
                    pdf_s3_path=pdf_s3_path,
                    page_num=page_num,
                    round=data["round"],
                    start_index=start_index,  # Byte offset in the original file
                    length=line_length,       # Length in bytes
                    finish_reason="completion_error",
                    error=data.get("completion_error", None)
                ))
            else:
                # Try to parse the actual model response JSON
                assert "outputs" in data and len(data["outputs"]) > 0, "No outputs from model detected"

                try:
                    model_response_json = orjson.loads(data["outputs"][0]["text"])
                    page_response = PageResponse(**model_response_json)

                    last_error = data.get("completion_error", None)

                    if not page_response.is_rotation_valid:
                        last_error = "rotation_invalid"

                    index_entries.append(DatabaseManager.BatchInferenceRecord(
                        inference_s3_path=inference_s3_path,
                        pdf_s3_path=pdf_s3_path,
                        page_num=page_num,
                        round=data["round"],
                        start_index=start_index,  # Byte offset in the original file
                        length=line_length,       # Length in bytes
                        finish_reason=data["outputs"][0]["finish_reason"],
                        error=last_error,
                    ))
                except Exception as e:
                    error_type = type(e).__name__
                    index_entries.append(DatabaseManager.BatchInferenceRecord(
                        inference_s3_path=inference_s3_path,
                        pdf_s3_path=pdf_s3_path,
                        page_num=page_num,
                        round=data["round"],
                        start_index=start_index,  # Byte offset in the original file
                        length=line_length,       # Length in bytes
                        finish_reason=data["outputs"][0]["finish_reason"],
                        error=error_type,
                    ))

        except Exception as e:
            logger.exception(f"Error processing line in {inference_s3_path}: {e}")
            # Optionally, you might want to add an index entry indicating an error here

        start_index += line_length  # Increment by the number of bytes

    return index_entries


def get_pdf_num_pages(s3_path: str) -> Optional[int]:
    try:
        with tempfile.NamedTemporaryFile("wb+", suffix=".pdf") as tf:
            tf.write(get_s3_bytes(pdf_s3, s3_path))
            tf.flush()

            reader = PdfReader(tf.name)
            return reader.get_num_pages()
    except Exception as ex:
        logger.warning(f"Warning, could not add {s3_path} due to {ex}")

    return None


def _get_page_data(page_index_entries: List[DatabaseManager.BatchInferenceRecord]) -> List[PageResponse]:
    usable_page_data = [get_s3_bytes(workspace_s3, page.inference_s3_path,
                                     start_index=page.start_index,
                                     end_index=page.start_index + page.length - 1) for page in page_index_entries]

    usable_page_final_results = []
    for page_data in usable_page_data:
        data = orjson.loads(page_data)
        model_response_json = orjson.loads(data["outputs"][0]["text"])
        page_response = PageResponse(**model_response_json)
        usable_page_final_results.append(page_response)

    return usable_page_final_results


def build_pdf_queries(s3_workspace: str, pdf: DatabaseManager.PDFRecord, cur_round: int, target_longest_image_dim: int, target_anchor_text_len: int) -> list[dict]:
    db = DatabaseManager(s3_workspace, skip_init=True)

    existing_pages = db.get_index_entries(pdf.s3_path)
    new_queries = []

    # Shortcut out of downloading the actual PDF
    if set(page.page_num for page in existing_pages if page.is_usable()) == set(range(1, pdf.num_pages + 1)):
        return []

    try:
        with tempfile.NamedTemporaryFile("wb+", suffix=".pdf") as tf:
            tf.write(get_s3_bytes(pdf_s3, pdf.s3_path))
            tf.flush()

            for target_page_num in range(1, pdf.num_pages + 1):
                # Is there an existing page that has no error
                if any(page.is_usable() and page.page_num == target_page_num for page in existing_pages):
                    continue

                has_errored_previously = sum(page.page_num == target_page_num for page in existing_pages)

                if has_errored_previously:
                    # Retry the page at least one more time regularly
                    new_queries.append({**build_page_query(tf.name, pdf.s3_path, target_page_num, target_longest_image_dim, target_anchor_text_len), "round": cur_round})
                    
                    # If the rotation was previously invalid, then apply a rotation  
                    rotated_page_data = _get_page_data([page for page in existing_pages if page.page_num == target_page_num and page.error == "rotation_invalid"])
                    rotation_corrections = set(page_data.rotation_correction for page_data in rotated_page_data)
                    for correction in rotation_corrections:
                        new_queries.append({**build_page_query(tf.name, pdf.s3_path, target_page_num, target_longest_image_dim, target_anchor_text_len, image_rotation=correction), "round": cur_round})
                    
                    # TODO: Try to provide a smaller prompt hint if that was the error
                else:
                    new_queries.append({**build_page_query(tf.name, pdf.s3_path, target_page_num, target_longest_image_dim, target_anchor_text_len), "round": cur_round})
    except Exception as ex:
        logger.warning(f"Warning, could not get batch inferences lines for {pdf.s3_path} due to {ex}")

    return new_queries

def build_dolma_doc(s3_workspace: str, pdf: DatabaseManager.PDFRecord) -> Optional[dict]:
    db = DatabaseManager(s3_workspace, skip_init=True)
    existing_pages = db.get_index_entries(pdf.s3_path)
    document_text = ""
    last_page_start_index = 0
    pdf_page_spans = []

    # Error out quickly if this document cannot be assembled
    for target_page_num in range(1, pdf.num_pages + 1):
        usable_pages = [page for page in existing_pages if page.is_usable() and page.page_num == target_page_num]

        if len(usable_pages) == 0:
            return None
    
    for target_page_num in range(1, pdf.num_pages + 1):
        usable_pages = [page for page in existing_pages if page.is_usable() and page.page_num == target_page_num]
        usable_page_final_results = _get_page_data(usable_pages)

        # Sort the pages:
        # 1. Prefer pages with `is_rotation_valid` set to True.
        # 2. Within those, sort by the length of the `natural_text` in descending order.
        usable_page_final_results.sort(
            key=lambda page: (not page.is_rotation_valid, -len(page.natural_text or ""))
        )

        target_page_final_result = usable_page_final_results[0]
    
        if target_page_final_result.natural_text is not None:
            document_text += target_page_final_result.natural_text + "\n"

        pdf_page_spans.append([last_page_start_index, len(document_text), target_page_num])
        last_page_start_index = len(document_text)

    metadata = {
            "Source-File": pdf.s3_path,
            "pdf-total-pages": pdf.num_pages,
        }
    id_ = hashlib.sha1(document_text.encode()).hexdigest()

    dolma_doc = {
        "id": id_,
        "text": document_text,
        "source": "pdelfin",
        "added": datetime.datetime.now().strftime("%Y-%m-%d"),
        "created": datetime.datetime.now().strftime("%Y-%m-%d"),
        "metadata": metadata,
        "attributes": {
            "pdf_page_numbers": pdf_page_spans
        }
    }

    return dolma_doc

def mark_pdfs_done(s3_workspace: str, dolma_docs: list[dict]):
    db = DatabaseManager(s3_workspace, skip_init=True)
    db.update_pdf_statuses({doc["metadata"]["Source-File"]: "completed" for doc in dolma_docs})

def get_current_round(s3_workspace: str) -> int:
    path = s3_workspace[5:]
    bucket, _, prefix = path.partition('/')

    inference_inputs_prefix = posixpath.join(prefix, 'inference_inputs/')
    paginator = workspace_s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=inference_inputs_prefix, Delimiter='/')

    round_numbers = []
    for page in page_iterator:
        for common_prefix in page.get('CommonPrefixes', []):
            round_prefix = common_prefix.get('Prefix')
            # Extract 'round_X' from the prefix
            round_dir = posixpath.basename(posixpath.dirname(round_prefix))
            if round_dir.startswith('round_'):
                try:
                    round_num = int(round_dir[len('round_'):])
                    round_numbers.append(round_num)
                except ValueError:
                    pass
    if round_numbers:
        current_round = max(round_numbers) + 1
    else:
        current_round = 0
    return current_round


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manager for running millions of PDFs through a batch inference pipeline')
    parser.add_argument('workspace', help='The S3 path where work will be done e.g., s3://bucket/prefix/)')
    parser.add_argument('--add_pdfs', help='Path to add pdfs stored in s3 to the workspace, can be a glob path s3://bucket/prefix/*.pdf or path to file containing list of pdf paths', default=None)
    parser.add_argument('--target_longest_image_dim', type=int, help='Dimension on longest side to use for rendering the pdf pages', default=1024)
    parser.add_argument('--target_anchor_text_len', type=int, help='Maximum amount of anchor text to use (characters)', default=6000)
    parser.add_argument('--workspace_profile', help='S3 configuration profile for accessing the workspace', default=None)
    parser.add_argument('--pdf_profile', help='S3 configuration profile for accessing the raw pdf documents', default=None)
    parser.add_argument('--max_size_mb', type=int, default=250, help='Max file size in MB')
    parser.add_argument('--workers', type=int, help='Number of workers to run in the processpool')
    parser.add_argument('--reindex', action='store_true', default=False, help='Reindex all of the page_results')
    parser.add_argument('--skip_build_queries', action='store_true', default=False, help='Skip generation of new pdf page queries for batch inferencing')
    args = parser.parse_args()

    if args.workspace_profile:
        workspace_session = boto3.Session(profile_name=args.workspace_profile)
        workspace_s3 = workspace_session.client("s3")

    if args.pdf_profile:
        pdf_session = boto3.Session(profile_name=args.pdf_profile)
        pdf_s3 = pdf_session.client("s3")

    db = DatabaseManager(args.workspace)
    logger.info(f"Loaded db at {db.db_path}")

    if args.reindex:
        db.clear_index()
        logger.info("Cleared existing index.")

    current_round = get_current_round(args.workspace)
    logger.info(f"Current round is {current_round}")

    # One shared executor to rule them all
    executor = ProcessPoolExecutor(max_workers=args.workers)

    # If you have new PDFs, step one is to add them to the list
    if args.add_pdfs:
        if args.add_pdfs.startswith("s3://"):
            logger.info(f"Querying all PDFs at {args.add_pdfs}")
            
            all_pdfs = expand_s3_glob(pdf_s3, args.add_pdfs)
            print(f"Found {len(all_pdfs):,} total pdf paths")
        elif os.path.exists(args.add_pdfs):
            with open(args.add_pdfs, "r") as f:
                all_pdfs = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
        else:
            raise ValueError("add_pdfs argument needs to be either an s3 glob search path, or a local file contains pdf paths (one per line)")

        all_pdfs = [pdf for pdf in all_pdfs if not db.pdf_exists(pdf)]
        logger.info(f"Need to import {len(all_pdfs):,} total new pdf paths")

        future_to_path = {executor.submit(get_pdf_num_pages, s3_path): s3_path for s3_path in all_pdfs}
        for future in tqdm(as_completed(future_to_path), total=len(future_to_path), desc="Adding PDFs"):
            s3_path = future_to_path[future]
            num_pages = future.result()
            if num_pages and not db.pdf_exists(s3_path):
                db.add_pdf(s3_path, num_pages, "pending")

        logger.info("Completed adding new PDFs.")

    # Now build an index of all the pages that were processed within the workspace so far
    logger.info("Indexing all batch inference sent to this workspace")
    inference_output_paths = expand_s3_glob(workspace_s3, f"{args.workspace}/inference_outputs/*.jsonl")

    inference_output_paths = {
        s3_path: etag for s3_path, etag in inference_output_paths.items()
        if not db.is_file_processed(s3_path, etag)
    }

    logger.info(f"Found {len(inference_output_paths):,} new batch inference results to index")
    future_to_path = {executor.submit(process_jsonl_content, s3_path): (s3_path, etag) for s3_path, etag in inference_output_paths.items()}

    for future in tqdm(as_completed(future_to_path), total=len(future_to_path), desc="Indexing Inference Results"):
        s3_path, etag = future_to_path.pop(future)
        try:
            inference_records = future.result()

            db.delete_index_entries_by_inference_s3_path(s3_path)
            db.add_index_entries(inference_records)
            db.update_processed_file(s3_path, etag=etag)
        except urllib3.exceptions.SSLError:
            logger.warning(f"Cannot load inference file {s3_path} due to SSL error, will retry another time")
        except Exception as e:
            logger.exception(f"Failed to index inference file {s3_path}: {e}")

    # Now query each pdf, if you have all of the pages needed (all pages present, error is null and finish_reason is stop), then you assemble it into a dolma document and output it
    # If you don't have every page, or if you have pages with errors, then you output a new batch of inference items to use
    if db.get_last_indexed_round() < current_round - 1:
        logger.warning(f"WARNING: No new batch inference results found, you need to run batch inference on {args.workspace}/inference_inputs/round_{current_round - 1}")
        potentially_done_pdfs = db.get_pdfs_by_status("pending")
    elif args.skip_build_queries:
        logger.info(f"Skipping generating new batch inference files")
        potentially_done_pdfs = db.get_pdfs_by_status("pending")
    else:
        logger.info("Creating batch inference files for new PDFs")
        pdf_list = list(db.get_pdfs_by_status("pending"))
        pdf_iter = iter(pdf_list)
        pending_futures = {}
        potentially_done_pdfs = []
        lines_written = 0
        new_inference_writer = BatchWriter(f"{args.workspace}/inference_inputs/round_{current_round}", args.max_size_mb)
        total_pdfs = len(pdf_list)
        max_pending = 300

        with tqdm(total=total_pdfs, desc="Building Batch Queries") as pbar:
            # Submit initial batch of futures
            for _ in range(min(max_pending, total_pdfs)):
                pdf = next(pdf_iter)
                future = executor.submit(
                    build_pdf_queries, args.workspace, pdf, current_round, args.target_longest_image_dim,args.target_anchor_text_len,
                )
                pending_futures[future] = pdf

            while pending_futures:
                # Wait for the next future to complete
                done, _ = concurrent.futures.wait(
                    pending_futures.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                for future in done:
                    pdf = pending_futures.pop(future)
                    inference_lines = future.result()

                    if len(inference_lines) == 0:
                        potentially_done_pdfs.append(pdf)

                    for line in inference_lines:
                        lines_written += 1

                        if line is not None:
                            new_inference_writer.write_line(line)

                    pbar.update(1)

                    # Submit a new future if there are more PDFs
                    try:
                        pdf = next(pdf_iter)
                        future = executor.submit(
                            build_pdf_queries, args.workspace, pdf, current_round, args.target_longest_image_dim,args.target_anchor_text_len,
                        )
                        pending_futures[future] = pdf
                    except StopIteration:
                        pass  # No more PDFs to process

        new_inference_writer.close()

        if lines_written > 0:
            logger.info(f"Added {lines_written:,} new batch inference requests")

    # Now, finally, assemble any potentially done docs into dolma documents
    logger.info(f"Assembling potentially finished PDFs into Dolma documents at {args.workspace}/output")
    future_to_path = {executor.submit(build_dolma_doc, args.workspace, pdf): pdf for pdf in potentially_done_pdfs}
    new_output_writer = BatchWriter(f"{args.workspace}/output", args.max_size_mb, after_flush=partial(mark_pdfs_done, args.workspace))

    for future in tqdm(as_completed(future_to_path), total=len(future_to_path), desc="Assembling Dolma Docs"):
        pdf = future_to_path.pop(future)
        dolma_doc = future.result()
        
        if dolma_doc is not None:
            new_output_writer.write_line(dolma_doc)

    new_output_writer.close()

    logger.info("Final statistics:")

    # Output the number of PDFs in each status "pending" and "completed"
    pending_pdfs = db.get_pdfs_by_status("pending")
    completed_pdfs = db.get_pdfs_by_status("completed")

    logger.info(f"Pending PDFs: {len(pending_pdfs):,} ({sum(doc.num_pages for doc in pending_pdfs):,} pages)")
    logger.info(f"Completed PDFs: {len(completed_pdfs):,} ({sum(doc.num_pages for doc in completed_pdfs):,} pages)")

    # For each round, outputs a report of how many pages were processed, how many had errors, and a breakdown by (error, finish_reason)
    total_rounds = db.get_last_indexed_round() + 1
    for round_num in range(total_rounds):
        db.cursor.execute("""
            SELECT COUNT(*), error, finish_reason
            FROM page_results
            WHERE round = ?
            GROUP BY error, finish_reason
        """, (round_num,))
        
        results = db.cursor.fetchall()
        
        total_pages = sum(count for count, _, _ in results)
        logger.info(f"Inference Round {round_num} - {total_pages:,} pages processed:")

        for count, error, finish_reason in results:
            error_str = error if error is not None else "None"
            logger.info(f"  (error: {error_str}, finish_reason: {finish_reason}) -> {count:,} pages")

    logger.info("Work finished, waiting for all workers to finish cleaning up")
    executor.shutdown(wait=True)
    db.close()
