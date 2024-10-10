import os
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

    if not files:
        print("No .json or .jsonl files found in the specified S3 path.")
        return

    # Use ThreadPoolExecutor to process files with tqdm progress bar
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, s3, bucket, key, etag, db_path) for key, etag in files.items()]
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

def process_file(s3, bucket, key, etag, db_path):
    s3_path = f's3://{bucket}/{key}'
    try:
        # Connect to duckdb
        conn = duckdb.connect(database=db_path)
        cursor = conn.cursor()

        # Check if file has already been processed with the same ETag
        cursor.execute("SELECT etag FROM processed_files WHERE s3_path = ?", (key,))
        result = cursor.fetchone()

        if result and result[0] == etag:
            # File has already been processed with the same ETag
            # Optionally, log that the file was skipped
            # print(f"Skipping already processed file: {s3_path}")
            conn.close()
            return
        else:
            # Get the object
            obj = s3.get_object(Bucket=bucket, Key=key)

            # Read the content as bytes
            content = obj['Body'].read()

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
        print(f"Error processing file {s3_path}: {e}")

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
