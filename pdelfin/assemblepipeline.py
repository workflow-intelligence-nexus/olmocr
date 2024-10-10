import os
import sys
import hashlib
import boto3
import duckdb
import json
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def build_index(s3_path):
    # Hash the s3_path to get a cache key
    cache_key = hashlib.sha256(s3_path.encode('utf-8')).hexdigest()
    cache_dir = os.path.join('.cache', cache_key)
    os.makedirs(cache_dir, exist_ok=True)
    db_path = os.path.join(cache_dir, 'index.db')

    # Connect to duckdb and create tables if not exist
    print("Building page index at", db_path)
    conn = duckdb.connect(database=db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS index_table (
            custom_id TEXT,
            s3_path TEXT,
            start_index BIGINT,
            end_index BIGINT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processed_files (
            s3_path TEXT PRIMARY KEY,
            etag TEXT
        )
    """)
    conn.commit()
    conn.close()

    s3 = boto3.client('s3')
    bucket, prefix = parse_s3_path(s3_path)

    # List all .json and .jsonl files under s3_path with their ETags
    files = list_s3_files(s3, bucket, prefix)

    # Filter out files that have already been processed
    files_to_process = filter_processed_files(db_path, files)

    if not files_to_process:
        print("All files have been processed. Nothing to do.")
        return

    # Use ThreadPoolExecutor to process files with tqdm progress bar
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_file, s3, bucket, key, etag, db_path) for key, etag in files_to_process.items()]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            pass

def parse_s3_path(s3_path):
    if not s3_path.startswith('s3://'):
        raise ValueError('s3_path must start with s3://')
    path = s3_path[5:]
    bucket, _, prefix = path.partition('/')
    return bucket, prefix

def list_s3_files(s3, bucket, prefix):
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    files = {}
    for page in page_iterator:
        contents = page.get('Contents', [])
        for obj in contents:
            key = obj['Key']
            if key.endswith('.json') or key.endswith('.jsonl'):
                # Retrieve ETag for each file
                files[key] = obj['ETag'].strip('"')
    return files

def filter_processed_files(db_path, files):
    conn = duckdb.connect(database=db_path)
    cursor = conn.cursor()

    # Retrieve processed files
    cursor.execute("SELECT s3_path, etag FROM processed_files")
    processed = dict(cursor.fetchall())

    # Filter out files that are already processed with the same ETag
    files_to_process = {}
    for key, etag in files.items():
        if key not in processed or processed[key] != etag:
            files_to_process[key] = etag

    conn.close()
    return files_to_process

def process_file(s3, bucket, key, etag, db_path):
    try:
        # Get the object
        obj = s3.get_object(Bucket=bucket, Key=key)
        s3_path = f's3://{bucket}/{key}'

        # Read the content as bytes
        content = obj['Body'].read()

        # Connect to duckdb
        conn = duckdb.connect(database=db_path)
        cursor = conn.cursor()

        # Process the file as JSONL
        process_jsonl_content(content, s3_path, cursor)

        # Update the processed_files table
        cursor.execute("""
            INSERT INTO processed_files (s3_path, etag)
            VALUES (?, ?)
            ON CONFLICT (s3_path) DO UPDATE SET etag=excluded.etag
        """, (key, etag))

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error processing file {key}: {e}")

def process_jsonl_content(content, s3_path, cursor):
    start_index = 0
    lines = content.splitlines(keepends=True)
    for line in lines:
        line_length = len(line)
        end_index = start_index + line_length
        try:
            data = json.loads(line)
            custom_id = data.get('custom_id')
            if custom_id:
                cursor.execute("""
                    INSERT INTO index_table (custom_id, s3_path, start_index, end_index)
                    VALUES (?, ?, ?, ?)
                """, (custom_id, s3_path, start_index, end_index))
        except json.JSONDecodeError:
            pass  # Handle JSON decode errors if necessary
        start_index = end_index

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build a local index of JSON files from S3.')
    parser.add_argument('s3_path', help='The S3 path to process (e.g., s3://bucket/prefix/)')
    args = parser.parse_args()

    build_index(args.s3_path)
