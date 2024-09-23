import os
import glob
import random
import subprocess
import base64
import argparse
import boto3
import json
from openai import OpenAI
from pypdf import PdfReader
from tqdm import tqdm
from typing import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

# reuse mise pdf filtering base code
from pdelfin.filter import PdfFilter

TARGET_IMAGE_DIM = 2048

def _build_prompt(base_text: str) -> str:
    return (
        f"Below is the image of one page of a PDF document, as well as a the raw textual content that was previously extracted for it. "
        f"Just return the plain text representation of this document as if you were reading it naturally.\n"
        f"Turn equations into a LaTeX representation. Remove the headers and footers, but keep references and footnotes.\n"
        f"Read any natural handwriting.\n"
        f"If there is no text at all that you think you should read, just output [NO TEXT].\n"
        f"Do not hallucinate.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
    )

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pdf_filter = PdfFilter()

def build_page_query(local_pdf_path: str, pretty_pdf_path: str, page: int) -> dict:
    pdf = PdfReader(local_pdf_path)
    pdf_page = pdf.pages[page - 1]
    longest_dim = max(pdf_page.mediabox.width, pdf_page.mediabox.height)

    # Convert PDF page to PNG using pdftoppm
    pdftoppm_result = subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-f",
            str(page),
            "-l",
            str(page),
            "-r",
            str(TARGET_IMAGE_DIM * 72 / longest_dim),
            local_pdf_path,
        ],
        timeout=120,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert pdftoppm_result.returncode == 0, pdftoppm_result.stderr
    image_base64 = base64.b64encode(pdftoppm_result.stdout).decode("utf-8")

    # Extract text from the PDF page using pdftotext
    pdftotext_result = subprocess.run(
        ["pdftotext", "-f", str(page), "-l", str(page), local_pdf_path, "-"],
        timeout=60,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert pdftotext_result.returncode == 0
    base_text = pdftotext_result.stdout.decode("utf-8")

    # Construct OpenAI Batch API request format
    return {
        "custom_id": f"{pretty_pdf_path}-{page}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-2024-08-06",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _build_prompt(base_text)},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 3000
        }
    }

def sample_pdf_pages(num_pages: int, first_n_pages: int, max_sample_pages: int) -> list:
    if num_pages <= first_n_pages:
        return list(range(1, num_pages + 1))  # Return all pages if fewer than first_n_pages
    sample_pages = list(range(1, first_n_pages + 1))  # Always get the first_n_pages
    remaining_pages = list(range(first_n_pages + 1, num_pages + 1))
    if remaining_pages:
        sample_pages += random.sample(remaining_pages, min(max_sample_pages - first_n_pages, len(remaining_pages)))
    return sample_pages

def fetch_s3_file(s3_url: str, local_path: str) -> str:
    parsed = urlparse(s3_url)
    bucket_name = parsed.netloc
    key = parsed.path.lstrip('/')
    
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, key, local_path)
    return local_path

def process_pdf(pdf_path: str, first_n_pages: int, max_sample_pages: int) -> Generator[dict, None, None]:
    if pdf_path.startswith("s3://"):
        local_pdf_path = os.path.join("/tmp", os.path.basename(pdf_path))
        fetch_s3_file(pdf_path, local_pdf_path)
    else:
        local_pdf_path = pdf_path

    if pdf_filter.filter_out_pdf(local_pdf_path):
        print(f"Skipping {local_pdf_path} due to common filter")
        return []
    
    pretty_pdf_path = pdf_path

    pdf = PdfReader(local_pdf_path)
    num_pages = len(pdf.pages)
    
    sample_pages = sample_pdf_pages(num_pages, first_n_pages, max_sample_pages)
    
    result = []
    for page in sample_pages:
        try:
            query = build_page_query(local_pdf_path, pretty_pdf_path, page)
            result.append(query)
        except Exception as e:
            print(f"Error processing page {page} of {pdf_path}: {e}")

    return result

def main():
    parser = argparse.ArgumentParser(description="Sample PDFs and create requests for GPT-4o.")
    parser.add_argument("--glob_path", type=str, help="Local or S3 path glob (e.g., *.pdf or s3://bucket/pdfs/*.pdf).")
    parser.add_argument("--path_list", type=str, help="Path to a file containing paths to PDFs, one per line.")
    parser.add_argument("--num_sample_docs", type=int, default=5000, help="Number of PDF documents to sample.")
    parser.add_argument("--first_n_pages", type=int, default=0, help="Always sample the first N pages of each PDF.")
    parser.add_argument("--max_sample_pages", type=int, default=15, help="Max number of pages to sample per PDF.")
    parser.add_argument("--output", type=str, default="openai_batch_data", help="Output destination")
    args = parser.parse_args()

    # Load PDF paths from glob or path_list
    pdf_paths = []
    if args.glob_path:
        if args.glob_path.startswith("s3://"):
            # Handle S3 globbing using boto3
            parsed = urlparse(args.glob_path)
            s3 = boto3.client('s3')
            bucket_name = parsed.netloc
            prefix = os.path.dirname(parsed.path.lstrip('/')) + "/"
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            for obj in response.get('Contents', []):
                if obj['Key'].endswith('.pdf'):
                    pdf_paths.append(f"s3://{bucket_name}/{obj['Key']}")
        else:
            # Handle local globbing
            pdf_paths = glob.glob(args.glob_path)
    elif args.path_list:
        with open(args.path_list, 'r') as f:
            pdf_paths = [line.strip() for line in f]

    random.shuffle(pdf_paths)

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
    with ThreadPoolExecutor(max_workers=60) as executor:
        futures = []

        with tqdm(desc="Processing PDFs", leave=False, total=args.num_sample_docs) as pb:
            for pdf_path in pdf_paths:
                futures.append(executor.submit(process_pdf, pdf_path, args.first_n_pages, args.max_sample_pages))

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
