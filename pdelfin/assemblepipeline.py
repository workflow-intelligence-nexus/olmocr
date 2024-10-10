import os
import hashlib
import boto3
import sqlite3
import json
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def build_index(s3_path):
    # Hash the s3_path to get a cache key
    cache_key = hashlib.sha256(s3_path.encode('utf-8')).hexdigest()
    home_cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'pdelfin', cache_key)
    os.makedirs(home_cache_dir, exist_ok=True)
    db_path = os.path.join(home_cache_dir, 'index.db')

    # Connect to sqlite and create tables if not exist
    print("Building page index at", db_path)
    conn = sqlite3.connect(db_path)
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

    s3 = boto3.client('s3')
    bucket, prefix = parse_s3_path(s3_path)

    # List all .json and .jsonl files under s3_path with their ETags
    files = list_s3_files(s3, bucket, prefix)

    if not files:
        print("No .json or .jsonl files found in the specified S3 path.")
        return

    # Prepare a list of files that need processing
    files_to_process = []
    for key, etag in files.items():
        cursor.execute("SELECT etag FROM processed_files WHERE s3_path = ?", (key,))
        db_result = cursor.fetchone()
        if db_result and db_result[0] == etag:
            # File has already been processed with the same ETag
            pass  # Skip
        else:
            files_to_process.append((key, etag))

    if not files_to_process:
        print("All files are up to date. No processing needed.")
        return

    # Use ProcessPoolExecutor to process files with tqdm progress bar
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, bucket, key, etag) for key, etag in files_to_process]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            s3_path, key, etag, index_entries = future.result()
            if index_entries:
                cursor.executemany("""
                    INSERT INTO index_table (custom_id, s3_path, start_index, end_index)
                    VALUES (?, ?, ?, ?)
                """, index_entries)
            # Update the processed_files table
            cursor.execute("""
                INSERT INTO processed_files (s3_path, etag)
                VALUES (?, ?)
                ON CONFLICT(s3_path) DO UPDATE SET etag=excluded.etag
            """, (key, etag))
            conn.commit()

    conn.close()

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build a local index of JSON files from S3.')
    parser.add_argument('s3_path', help='The S3 path to process (e.g., s3://bucket/prefix/)')
    args = parser.parse_args()

    # Step one, build an index of all the pages that were processed
    build_index(args.s3_path)

    
