import os
import glob
import random
import subprocess
import base64
import argparse
import boto3
import json
from pypdf import PdfReader
from tqdm import tqdm
from typing import Generator
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from urllib.parse import urlparse

from pdelfin.data.renderpdf import render_pdf_to_base64png
from pdelfin.prompts import build_finetuning_prompt
from pdelfin.prompts.anchor import get_anchor_text
from pdelfin.filter import PdfFilter


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

def process_pdf(pdf_path: str, first_n_pages: int, max_sample_pages: int, no_filter: bool) -> Generator[dict, None, None]:
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

def main():
    parser = argparse.ArgumentParser(description="Given a bunch of PDFs, prepares a mise/birr workflow to run them through a conversion mechanism")
    parser.add_argument("--glob_path", type=str, help="Local or S3 path glob (e.g., *.pdf or s3://bucket/pdfs/*.pdf).")
    parser.add_argument("--path_list", type=str, help="Path to a file containing paths to PDFs, one per line.")
    parser.add_argument("--max_size_mb", type=int, default=250, help="Max number of mb's of entries to put in each birr workitem")
    parser.add_argument("--no_filter", action="store_true", help="Disables the basic spam/language filtering so that ALL pdfs listed are used")
    parser.add_argument("--output", type=str, default="mise_batch_data", help="Output destination")
    args = parser.parse_args()

    pdf_paths = []

    # Load PDF paths from glob or path_list using reservoir sampling
    if args.glob_path:
        if args.glob_path.startswith("s3://"):
            # Handle S3 globbing using boto3 with pagination
            parsed = urlparse(args.glob_path)
            s3 = boto3.client('s3')
            bucket_name = parsed.netloc
            prefix = os.path.dirname(parsed.path.lstrip('/')) + "/"
            paginator = s3.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

            for page in page_iterator:
                for obj in page.get('Contents', []):
                    if obj['Key'].endswith('.pdf'):
                        pdf_paths.append(f"s3://{bucket_name}/{obj['Key']}")

        else:
            # Handle local globbing using glob.iglob()
            for path in glob.iglob(args.glob_path, recursive=True):
                pdf_paths.append(path)
    elif args.path_list:
        with open(args.path_list, 'r') as f:
            for line in f:
                n += 1
                path = line.strip()
                pdf_paths.append(path)


    print(f"Loaded and shuffled {len(pdf_paths)} paths to use.")

    # Rest of the code remains the same
    cur_file_num = 0
    output_dir = args.output
    max_file_size = 99 * 1024 * 1024  # 99MB in bytes
    cur_file_size = 0
    cur_file_path = os.path.join(output_dir, f"output_{cur_file_num}.jsonl")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the first file for writing
    cur_file = open(cur_file_path, 'w')

    # Counter to track PDFs that produce at least one output
    pdfs_with_output = 0

    # Using ThreadPoolExecutor to process files concurrently
    with ProcessPoolExecutor() as executor:
        futures = []

        with tqdm(desc="Processing PDFs", leave=False, total=args.num_sample_docs) as pb:
            for pdf_path in pdf_paths:
                futures.append(executor.submit(process_pdf, pdf_path, args.first_n_pages, args.max_sample_pages, args.no_filter))

            for future in as_completed(futures):
                has_output = False  # Track if the current PDF produces at least one request
                try:
                    request_results = future.result()  # Get the result from the thread

                    for request_obj in request_results:
                        request_json = json.dumps(request_obj)
                        request_size = len(request_json.encode('utf-8'))  # Calculate size in bytes

                        # Check if the current request can fit in the current file
                        if cur_file_size + request_size > max_file_size:
                            # Close the current file and create a new one
                            cur_file.close()
                            cur_file_num += 1
                            cur_file_path = os.path.join(output_dir, f"output_{cur_file_num}.jsonl")
                            cur_file = open(cur_file_path, 'w')
                            cur_file_size = 0  # Reset file size

                        # Write the JSON entry to the file
                        cur_file.write(request_json)
                        cur_file.write("\n")
                        cur_file_size += request_size

                        has_output = True  # At least one request object was generated

                    if has_output:
                        pdfs_with_output += 1
                        pb.update(1)

                        if pdfs_with_output >= args.num_sample_docs:
                            executor.shutdown(cancel_futures=True)
                            break

                except Exception as e:
                    print(f"Error processing {pdf_path}: {str(e)}")

    # Close the last open file
    cur_file.close()

    # Print or log the number of PDFs that resulted in at least one output
    print(f"Number of sampled PDFs that produced at least one output: {pdfs_with_output}")

if __name__ == "__main__":
    main()
