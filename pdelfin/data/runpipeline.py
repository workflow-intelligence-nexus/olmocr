import os
import glob
import random
import subprocess
import base64
import argparse
import boto3
import json
import hashlib
from pypdf import PdfReader
from tqdm import tqdm
from typing import Generator, List
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from urllib.parse import urlparse

from pdelfin.data.renderpdf import render_pdf_to_base64png
from pdelfin.prompts import build_finetuning_prompt
from pdelfin.prompts.anchor import get_anchor_text
from pdelfin.filter import PdfFilter

import logging
import smart_open
import posixpath  # Import posixpath for S3 path handling

logging.getLogger("pypdf").setLevel(logging.ERROR)

pdf_filter = PdfFilter()

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
        "temperature": 0.1,
        "max_tokens": 6000,
    }

def fetch_s3_file(s3_url: str, local_path: str) -> str:
    parsed = urlparse(s3_url)
    bucket_name = parsed.netloc
    key = parsed.path.lstrip('/')

    s3 = boto3.client('s3')
    s3.download_file(bucket_name, key, local_path)
    return local_path

def process_pdf(pdf_path: str, no_filter: bool) -> List[dict]:
    if pdf_path.startswith("s3://"):
        local_pdf_path = os.path.join("/tmp", os.path.basename(pdf_path))
        fetch_s3_file(pdf_path, local_pdf_path)
    else:
        local_pdf_path = pdf_path

    if (not no_filter) and pdf_filter.filter_out_pdf(local_pdf_path):
        print(f"Skipping {local_pdf_path} due to common filter")
        return []

    pretty_pdf_path = pdf_path

    pdf = PdfReader(local_pdf_path)
    num_pages = len(pdf.pages)

    sample_pages = list(range(1, num_pages + 1))
    result = []
    for page in sample_pages:
        try:
            query = build_page_query(local_pdf_path, pretty_pdf_path, page)
            result.append(query)
        except Exception as e:
            print(f"Error processing page {page} of {pdf_path}: {e}")

    return result

def is_glob_pattern(path: str) -> bool:
    return any(char in path for char in ['*', '?', '[', ']'])

def expand_s3_glob(s3_glob: str) -> list:
    parsed = urlparse(s3_glob)
    bucket_name = parsed.netloc
    prefix = os.path.dirname(parsed.path.lstrip('/')).rstrip('/') + "/"
    pattern = os.path.basename(parsed.path)

    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    matched_files = []
    for page in page_iterator:
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.pdf') and glob.fnmatch.fnmatch(key, posixpath.join(prefix, pattern)):
                matched_files.append(f"s3://{bucket_name}/{key}")

    return matched_files

def compute_hash(content: str) -> str:
    """Compute a 20-character SHA1 hash of the given content."""
    sha1 = hashlib.sha1()
    sha1.update(content.encode('utf-8'))
    return sha1.hexdigest()[:20]

def get_smart_open_write_path(output_path: str, hash_str: str) -> str:
    """Generate the full output path with hash in the filename."""
    parsed = urlparse(output_path)
    if parsed.scheme in ('s3', 's3a', 's3n'):
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        # Ensure the key is treated as a directory by appending a slash if not present
        if key and not key.endswith('/'):
            key += '/'
        # Use posixpath to correctly join S3 paths
        full_key = posixpath.join(key, f"output_{hash_str}.jsonl")
        return f"s3://{bucket}/{full_key}"
    else:
        dir_path = output_path
        filename = f"output_{hash_str}.jsonl"
        return os.path.join(dir_path, filename)

def main():
    parser = argparse.ArgumentParser(
        description="Given a bunch of PDFs, prepares a mise/birr workflow to run them through a conversion mechanism"
    )
    parser.add_argument(
        "pdf_paths",
        nargs='*',
        help=(
            "List of PDF paths to process. If a single argument contains glob patterns (e.g., *.pdf or s3://bucket/pdfs/*.pdf), "
            "it will be expanded accordingly."
        )
    )
    parser.add_argument(
        "--path_list",
        type=str,
        help="Path to a file containing paths to PDFs, one per line."
    )
    parser.add_argument(
        "--max_size_mb",
        type=int,
        default=250,
        help="Max number of MBs of entries to put in each birr workitem"
    )
    parser.add_argument(
        "--no_filter",
        action="store_true",
        help="Disables the basic spam/language filtering so that ALL pdfs listed are used"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mise_batch_data",
        help="Output destination (can be a local path or an S3 URI)"
    )
    args = parser.parse_args()

    pdf_paths = []

    # Load PDF paths from positional arguments or path_list
    if args.pdf_paths:
        if len(args.pdf_paths) == 1 and is_glob_pattern(args.pdf_paths[0]):
            glob_path = args.pdf_paths[0]
            if glob_path.startswith("s3://"):
                # Handle S3 globbing
                expanded_paths = expand_s3_glob(glob_path)
                pdf_paths.extend(expanded_paths)
            else:
                # Handle local filesystem globbing
                expanded_paths = glob.glob(glob_path, recursive=True)
                pdf_paths.extend(expanded_paths)
        else:
            # Treat positional arguments as list of PDF paths
            pdf_paths.extend(args.pdf_paths)

    if args.path_list:
        with open(args.path_list, 'r') as f:
            for line in f:
                path = line.strip()
                if path:
                    pdf_paths.append(path)

    # Remove duplicates and shuffle
    pdf_paths = list(set(pdf_paths))
    random.shuffle(pdf_paths)

    print(f"Loaded and shuffled {len(pdf_paths)} paths to use.")

    # Prepare for output
    output_dir = args.output
    max_file_size = args.max_size_mb * 1024 * 1024  # Convert MB to bytes

    # Determine if output is S3
    parsed_output = urlparse(output_dir)
    is_s3 = parsed_output.scheme in ('s3', 's3a', 's3n')

    # Initialize variables for batching
    batch = []
    batch_size = 0
    pdfs_with_output = 0

    # Function to write a batch
    def write_batch(batch: List[dict]):
        nonlocal output_dir
        if not batch:
            return
        batch_content = "\n".join(json.dumps(entry) for entry in batch) + "\n"
        hash_str = compute_hash(batch_content)
        output_path_with_hash = get_smart_open_write_path(output_dir, hash_str)
        with smart_open.open(output_path_with_hash, 'w') as f_out:
            f_out.write(batch_content)
        print(f"Wrote batch to {output_path_with_hash}")

    # Using ProcessPoolExecutor to process files concurrently
    with ProcessPoolExecutor() as executor:
        futures = []

        with tqdm(desc="Processing PDFs", leave=False, total=len(pdf_paths)) as pb:
            for pdf_path in pdf_paths:
                futures.append(executor.submit(process_pdf, pdf_path, args.no_filter))

            for future in as_completed(futures):
                try:
                    request_results = future.result()  # Get the result from the process

                    if request_results:
                        pdfs_with_output += 1  # Increment if there's at least one result

                    for request_obj in request_results:
                        request_json = json.dumps(request_obj)
                        request_size = len(request_json.encode('utf-8')) + 1  # +1 for newline

                        # Check if adding this entry would exceed the max size
                        if batch_size + request_size > max_file_size:
                            # Write the current batch
                            write_batch(batch)
                            # Reset the batch
                            batch = []
                            batch_size = 0

                        # Add the entry to the batch
                        batch.append(request_obj)
                        batch_size += request_size

                    pb.update(1)
                except Exception as e:
                    print(f"Error processing a PDF: {str(e)}")

    # Write any remaining batch
    write_batch(batch)

    # Print the number of PDFs that resulted in at least one output
    print(f"Number of sampled PDFs that produced at least one output: {pdfs_with_output}")

if __name__ == "__main__":
    main()
