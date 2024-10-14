import os
import hashlib
import boto3
import sqlite3
import json
import argparse
import glob
import tempfile
import posixpath

from dataclasses import dataclass
from pypdf import PdfReader
from tqdm import tqdm
from typing import Optional, List, Tuple, Dict
from urllib.parse import urlparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from pdelfin.data.renderpdf import render_pdf_to_base64png
from pdelfin.prompts import build_finetuning_prompt
from pdelfin.prompts.anchor import get_anchor_text

# Global s3 client for the whole script, feel free to adjust params if you need it
s3 = boto3.client('s3')

class DatabaseManager:
    @dataclass(frozen=True)
    class BatchInferenceRecord:
        s3_path: str
        page_num: int  # 1 indexed!
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

    def __init__(self, s3_workspace: str):
        cache_key = hashlib.sha256(s3_workspace.strip().lower().encode('utf-8')).hexdigest()
        home_cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'pdelfin', cache_key)
        os.makedirs(home_cache_dir, exist_ok=True)
        self.db_path = os.path.join(home_cache_dir, 'index.db')
        
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._initialize_tables()

    def _initialize_tables(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS page_results (
                s3_path TEXT,
                page_num INTEGER,
                start_index BIGINT,
                length BIGINT,
                finish_reason TEXT,
                error TEXT
            )
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_path ON page_results(s3_path)
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

    def get_current_round(self):
        round_value = self.get_metadata("round")
        return int(round_value) if round_value else 0

    def is_file_processed(self, s3_path, etag):
        self.cursor.execute("SELECT etag FROM processed_files WHERE s3_path = ?", (s3_path,))
        result = self.cursor.fetchone()
        return result is not None and result[0] == etag

    def add_index_entries(self, index_entries: List[BatchInferenceRecord]):
        if index_entries:
            self.cursor.executemany("""
                INSERT INTO page_results (s3_path, page_num, start_index, length, finish_reason, error)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [(entry.s3_path, entry.page_num, entry.start_index, entry.length, entry.finish_reason, entry.error) for entry in index_entries])
            self.conn.commit()

    def get_index_entries(self, s3_path: str) -> List[BatchInferenceRecord]:
        self.cursor.execute("""
            SELECT s3_path, page_num, start_index, length, finish_reason, error
            FROM page_results
            WHERE s3_path = ?
            ORDER BY page_num ASC
        """, (s3_path,))
        
        rows = self.cursor.fetchall()
        
        return [
            self.BatchInferenceRecord(
                s3_path=row[0],
                page_num=row[1],
                start_index=row[2],
                length=row[3],
                finish_reason=row[4],
                error=row[5]
            )
            for row in rows
        ]

    def update_processed_file(self, s3_path, etag):
        self.cursor.execute("""
            INSERT INTO processed_files (s3_path, etag)
            VALUES (?, ?)
            ON CONFLICT(s3_path) DO UPDATE SET etag=excluded.etag
        """, (s3_path, etag))
        self.conn.commit()

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
            print(f"PDF with s3_path '{s3_path}' already exists.")

    def get_pdf(self, s3_path: str) -> Optional[PDFRecord]:
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
    
    def get_pdfs_by_status(self, status: str) -> List[PDFRecord]:
        self.cursor.execute("""
            SELECT s3_path, num_pages, status
            FROM pdfs
            WHERE status == ?
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





def parse_s3_path(s3_path):
    if not s3_path.startswith('s3://'):
        raise ValueError('s3_path must start with s3://')
    path = s3_path[5:]
    bucket, _, prefix = path.partition('/')
    return bucket, prefix

def expand_s3_glob(s3_glob: str) -> Dict[str, str]:
    parsed = urlparse(s3_glob)
    bucket_name = parsed.netloc
    prefix = os.path.dirname(parsed.path.lstrip('/')).rstrip('/') + "/"
    pattern = os.path.basename(parsed.path)

    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    matched_files = {}
    for page in page_iterator:
        for obj in page.get('Contents', []):
            key = obj['Key']
            if glob.fnmatch.fnmatch(key, posixpath.join(prefix, pattern)):
                matched_files[f"s3://{bucket_name}/{key}"] = obj['ETag'].strip('"')

    return matched_files

def build_page_query(local_pdf_path: str, pretty_pdf_path: str, page: int) -> dict:
    image_base64 = render_pdf_to_base64png(local_pdf_path, page, 1024)
    anchor_text = get_anchor_text(local_pdf_path, page, pdf_engine="pdfreport")

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
        "temperature": 0.8,
        "max_tokens": 6000,
    }


def get_s3_bytes(s3_path: str, start_index: Optional[int] = None, end_index: Optional[int] = None) -> bytes:
    bucket, key = parse_s3_path(s3_path)

    # Build the range header if start_index and/or end_index are specified
    range_header = None
    if start_index is not None or end_index is not None:
        range_value = f"bytes={start_index or 0}-"
        if end_index is not None:
            range_value += str(end_index)
        range_header = {'Range': range_value}

    if range_header:
        obj = s3.get_object(Bucket=bucket, Key=key, Range=range_header['Range'])
    else:
        obj = s3.get_object(Bucket=bucket, Key=key)

    return obj['Body'].read()


def parse_custom_id(custom_id: str) -> Tuple[str, int]:
    s3_path = custom_id[:custom_id.rindex("-")]
    page_num = int(custom_id[custom_id.rindex("-") + 1:])
    return s3_path, page_num

def process_jsonl_content(s3_path) -> List[DatabaseManager.BatchInferenceRecord]:
    content = get_s3_bytes(s3_path).decode("utf-8")

    start_index = 0
    index_entries = []
    lines = content.splitlines(keepends=True)
    for line in lines:
        line_length = len(line)

        try:
            data = json.loads(line)
            s3_path, page_num = parse_custom_id(data["custom_id"])

            assert "outputs" in data and len(data["outputs"]) > 0, "No outputs from model detected"

            index_entries.append(DatabaseManager.BatchInferenceRecord(
                s3_path=s3_path,
                page_num=page_num,
                start_index=start_index,
                length=line_length,
                finish_reason=data["outputs"][0]["finish_reason"],
                error=data.get("completion_error", None)
            ))
        except json.JSONDecodeError:
            pass  # Handle JSON decode errors if necessary
        except Exception as e:
            print(f"Error processing line: {e}")

        start_index += line_length

    return index_entries

def get_pdf_num_pages(s3_path: str) -> Optional[int]:
    try:
        with tempfile.NamedTemporaryFile("wb+", suffix=".pdf") as tf:
            tf.write(get_s3_bytes(s3_path))
            tf.flush()

            reader = PdfReader(tf.name)
            return reader.get_num_pages()
    except Exception as ex:
        print(f"Warning, could not add {s3_path} due to {ex}")

    return None

def build_pdf_queries(s3_workspace: str, pdf: DatabaseManager.PDFRecord) -> list[dict]:
    db = DatabaseManager(s3_workspace)

    existing_pages = db.get_index_entries(pdf.s3_path)
    new_queries = []
    
    try:
        with tempfile.NamedTemporaryFile("wb+", suffix=".pdf") as tf:
            tf.write(get_s3_bytes(pdf.s3_path))
            tf.flush()

            for page in range(1, pdf.num_pages + 1):
                # Is there an existing page that has no error
                if any(page.is_usable() for page in existing_pages):
                    continue

                # TODO: Later, you may want to retry with different sampling parameters or do something else
                new_queries.append(build_page_query(tf.name, pdf.s3_path, page))
    except Exception as ex:
        print(f"Warning, could not get batch inferences lines for {pdf.s3_path} due to {ex}")

    return new_queries

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manager for running millions of PDFs through a batch inference pipeline')
    parser.add_argument('workspace', help='The S3 path where work will be done e.g., s3://bucket/prefix/)')
    parser.add_argument('--add_pdfs', help='Glob path to add PDFs (s3) to the workspace', default=None)
    parser.add_argument('--file_size_limit', type=int, default=250, help='Max file size in MB')
    args = parser.parse_args()

    db = DatabaseManager(args.workspace)
    print(f"Loaded db at {db.db_path}")
    print(f"Current round is {db.get_current_round()}\n")

    # One shared executor to rule them all
    executor = ProcessPoolExecutor()

    # If you have new PDFs, step one is to add them to the list
    if args.add_pdfs:
        assert args.add_pdfs.startswith("s3://"), "PDFs must live on s3"
        
        print(f"Querying all PDFs at {args.add_pdfs}")
        
        all_pdfs = expand_s3_glob(args.add_pdfs)
        print(f"Found {len(all_pdfs)} total pdf paths")

        all_pdfs = [pdf for pdf in all_pdfs if not db.pdf_exists(pdf)]
        print(f"Need to import {len(all_pdfs)} total new pdf paths")

        future_to_path = {executor.submit(get_pdf_num_pages, s3_path): s3_path for s3_path in all_pdfs}
        for future in tqdm(as_completed(future_to_path), total=len(future_to_path)):
            s3_path = future_to_path[future]
            num_pages = future.result()
            if num_pages and not db.pdf_exists(s3_path):
                db.add_pdf(s3_path, num_pages, "pending")

        print("\n")

    # Now build an index of all the pages that were processed within the workspace so far
    print("Indexing all batch inference sent to this workspace")
    inference_output_paths = expand_s3_glob(f"{args.workspace}/inference_outputs/*.jsonl")

    inference_output_paths = [
        (s3_path, etag) for s3_path, etag in inference_output_paths.items()
        if not db.is_file_processed(s3_path, etag)
    ]

    print(f"Found {len(inference_output_paths)} new batch inference results to index")

    future_to_path = {executor.submit(process_jsonl_content, s3_path): (s3_path, etag) for s3_path, etag in inference_output_paths}

    for future in tqdm(as_completed(future_to_path), total=len(future_to_path)):
        s3_path, etag = future_to_path[future]
        inference_lines = future.result()

        db.add_index_entries(inference_lines)
        db.update_processed_file(s3_path, etag=etag)

    # Now query each pdf, if you have all of the pages needed (all pages present, error is null and finish_reason is stop), then you assemble it into a dolma document and output it
    # If you don't have every page, or if you have pages with errors, then you output a new batch of inference items to use
    future_to_path = {executor.submit(build_pdf_queries, args.workspace, pdf): pdf for pdf in db.get_pdfs_by_status("pending")}

    for future in tqdm(as_completed(future_to_path), total=len(future_to_path)):
        pdf = future_to_path[future]
        inference_lines = future.result()





    # TODO
    # 1. build a class that will manage taking in dicts and outputting them as jsonls of up to the max size to the bucket
    # you'll need one for new batch inference lines, and one for finished dolma docs
    # 2. Have a way to apply basic spam + language filter if you can during add pdfs step
    # 3. For retrying, make it so you retry several times with different sampling parameters