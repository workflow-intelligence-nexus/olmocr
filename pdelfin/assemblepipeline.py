import os
import hashlib
import boto3
import sqlite3
import json
import argparse
import glob
import tempfile
import posixpath

from pypdf import PdfReader
from tqdm import tqdm
from typing import Optional
from urllib.parse import urlparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# Global s3 client for the whole script, feel free to adjust params if you need it
s3 = boto3.client('s3')

class DatabaseManager:
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
            CREATE TABLE IF NOT EXISTS index_table (
                custom_id TEXT,
                s3_path TEXT,
                start_index BIGINT,
                end_index BIGINT
            )
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_custom_id ON index_table(custom_id)
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
        self.cursor.execute("SELECT value FROM metadata WHERE key='round'")
        if self.cursor.fetchone() is None:
            self.cursor.execute("INSERT INTO metadata (key, value) VALUES ('round', '0')")
        self.conn.commit()

    def get_current_round(self):
        self.cursor.execute("SELECT value FROM metadata WHERE key='round'")
        result = self.cursor.fetchone()
        return int(result[0])

    def is_file_processed(self, s3_path, etag):
        self.cursor.execute("SELECT etag FROM processed_files WHERE s3_path = ?", (s3_path,))
        result = self.cursor.fetchone()
        return result is not None and result[0] == etag

    def add_index_entries(self, index_entries):
        if index_entries:
            self.cursor.executemany("""
                INSERT INTO index_table (custom_id, s3_path, start_index, end_index)
                VALUES (?, ?, ?, ?)
            """, index_entries)
            self.conn.commit()

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

    def get_pdf_status(self, s3_path: str) -> Optional[str]:
        self.cursor.execute("SELECT status FROM pdfs WHERE s3_path = ?", (s3_path,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def close(self):
        self.conn.close()

def build_index(s3_path):
    db_manager = DatabaseManager(s3_path)

    bucket, prefix = parse_s3_path(s3_path)

    # List all .json and .jsonl files under s3_path with their ETags
    files = expand_s3_glob(s3_path)

    if not files:
        print("No .json or .jsonl files found in the specified S3 path.")
        db_manager.close()
        return

    # Prepare a list of files that need processing
    files_to_process = [
        (key, etag) for key, etag in files.items()
        if not db_manager.is_file_processed(key, etag)
    ]

    if not files_to_process:
        print("All files are up to date. No processing needed.")
        db_manager.close()
        return

    # Use ProcessPoolExecutor to process files with tqdm progress bar
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_file, bucket, key, etag) 
            for key, etag in files_to_process
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            s3_path, key, etag, index_entries = future.result()
            if index_entries:
                db_manager.add_index_entries(index_entries)
            # Update the processed_files table
            db_manager.update_processed_file(key, etag)

    db_manager.close()

def parse_s3_path(s3_path):
    if not s3_path.startswith('s3://'):
        raise ValueError('s3_path must start with s3://')
    path = s3_path[5:]
    bucket, _, prefix = path.partition('/')
    return bucket, prefix

def expand_s3_glob(s3_glob: str) -> dict[str, str]:
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

def process_file(bucket, key, etag):
    s3 = boto3.client('s3')  # Initialize s3 client in the worker process
    s3_path = f's3://{bucket}/{key}'
    try:
        # Get the object
        obj = s3.get_object(Bucket=bucket, Key=key)
        # Read the content as bytes
        content = obj['Body'].read()
        # Process the file as JSONL
        index_entries = process_jsonl_content(content, s3_path)
        # Return the necessary data to the main process
        return s3_path, key, etag, index_entries
    except Exception as e:
        print(f"Error processing file {s3_path}: {e}")
        return s3_path, key, etag, []

def process_jsonl_content(content, s3_path):
    start_index = 0
    index_entries = []
    lines = content.splitlines(keepends=True)
    for line in lines:
        line_length = len(line)
        end_index = start_index + line_length
        try:
            data = json.loads(line)
            custom_id = data.get('custom_id')
            if custom_id:
                index_entries.append((custom_id, s3_path, start_index, end_index))
        except json.JSONDecodeError:
            pass  # Handle JSON decode errors if necessary
        start_index = end_index
    return index_entries

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
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manager for running millions of PDFs through a batch inference pipeline')
    parser.add_argument('workspace', help='The S3 path where work will be done e.g., s3://bucket/prefix/)')
    parser.add_argument('--pdfs', help='Glob path to PDFs (local or s3)', default=None)
    parser.add_argument('--file_size_limit', type=int, default=250, help='Max file size in MB')
    args = parser.parse_args()

    db = DatabaseManager(args.workspace)
    print(f"Loaded db at {db.db_path}")
    print(f"Current round is {db.get_current_round()}\n")

    # One shared executor to rule them all
    executor = ProcessPoolExecutor()

    # If you have new PDFs, add them to the list
    if args.pdfs:
        assert args.pdfs.startswith("s3://"), "PDFs must live on s3"
        
        print(f"Querying all PDFs at {args.pdfs}")
        
        all_pdfs = expand_s3_glob(args.pdfs)
        print(f"Found {len(all_pdfs)} total pdf paths")

        all_pdfs = [pdf for pdf in all_pdfs if not db.pdf_exists(pdf)]
        print(f"Need to import {len(all_pdfs)} total new pdf paths")

        future_to_path = {executor.submit(get_pdf_num_pages, s3_path): s3_path for s3_path in all_pdfs}
        for future in tqdm(as_completed(future_to_path), total=len(future_to_path)):
            s3_path = future_to_path[future]
            if future.result() and not db.pdf_exists(s3_path):
                db.add_pdf(s3_path, future.result(), "pending")

        print("\n")


    # Now build an index of all the pages that were processed within the workspace so far
    build_index(f"{args.workspace}/*.jsonl")

    # Now, for each pending book, find all pages which still need to be processed
    # and add them to the next round's batch inference jobs
    