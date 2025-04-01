# This script takes as arguments a olmocr workspace
# It will then go and list the s3://workspace/results folder and find all the output jsons in there
# Then, pick a random one, from there, read a random line
# Now, use render_pdf_to_base64webp to render that at 2048 resolution
# Now, do  30 of those (args.pages_per_output) and output them into a single nice looking static html file
# Also take an argument args.repeats which will repeat this whole process N times

import argparse
import datetime
import json
import os
import random
import re
import sqlite3
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import boto3
from tqdm import tqdm

from olmocr.data.renderpdf import render_pdf_to_base64webp
from olmocr.s3_utils import get_s3_bytes, parse_s3_path


def parse_args():
    parser = argparse.ArgumentParser(description="Scan OLMO OCR workspace results and create visual samples")
    parser.add_argument("workspace", help="OLMO OCR workspace path (s3://bucket/workspace)")
    parser.add_argument("--pages_per_output", type=int, default=30, help="Number of pages per output file")
    parser.add_argument("--repeats", type=int, default=1, help="Number of output files to generate")
    parser.add_argument("--pdf_profile", help="AWS profile for accessing PDFs")
    parser.add_argument("--output_dir", default="dolma_samples", help="Directory to save output HTML files")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker threads")
    parser.add_argument(
        "--db_path",
        default="~/s2pdf_url_data/d65142df-6588-4b68-a12c-d468b3761189.csv.db",
        help="Path to the SQLite database containing PDF hash to URL mapping",
    )
    return parser.parse_args()


def parse_pdf_hash(pretty_pdf_path: str) -> Optional[str]:
    pattern = r"s3://ai2-s2-pdfs/([a-f0-9]{4})/([a-f0-9]+)\.pdf"
    match = re.match(pattern, pretty_pdf_path)
    if match:
        return match.group(1) + match.group(2)
    return None


def get_original_url(pdf_hash: str, db_path: str) -> Optional[str]:
    """Look up the original URL for a PDF hash in the SQLite database."""
    if not pdf_hash:
        return None

    try:
        sqlite_db_path = os.path.expanduser(db_path)
        if not os.path.exists(sqlite_db_path):
            print(f"SQLite database not found at {sqlite_db_path}")
            return None

        conn = sqlite3.connect(sqlite_db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT uri FROM pdf_mapping WHERE pdf_hash = ?", (pdf_hash,))
        result = cursor.fetchone()

        conn.close()

        if result:
            return result[0]
        return None
    except Exception as e:
        print(f"Error looking up URL for PDF hash {pdf_hash}: {e}")
        return None


def list_result_files(s3_client, workspace_path):
    """List all JSON result files in the workspace results directory."""
    bucket, prefix = parse_s3_path(workspace_path)
    results_prefix = os.path.join(prefix, "results").rstrip("/") + "/"

    all_files = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=results_prefix):
        if "Contents" in page:
            all_files.extend([f"s3://{bucket}/{obj['Key']}" for obj in page["Contents"] if obj["Key"].endswith(".jsonl") or obj["Key"].endswith(".json")])

        if len(all_files) > 1000:
            break

    return all_files


def get_random_pages(s3_client, result_files, count=30):
    """Get random pages from the result files."""
    random_pages = []

    # Try to collect the requested number of pages
    attempts = 0
    max_attempts = count * 3  # Allow extra attempts to handle potential failures

    while len(random_pages) < count and attempts < max_attempts:
        attempts += 1

        # Pick a random result file
        if not result_files:
            print("No result files found!")
            break

        result_file = random.choice(result_files)

        try:
            # Get the content of the file
            content = get_s3_bytes(s3_client, result_file)
            lines = content.decode("utf-8").strip().split("\n")

            if not lines:
                continue

            # Pick a random line (which contains a complete document)
            line = random.choice(lines)
            doc = json.loads(line)

            # A Dolma document has "text", "metadata", and "attributes" fields
            if "text" not in doc or "metadata" not in doc or "attributes" not in doc:
                print(f"Document in {result_file} is not a valid Dolma document")
                continue

            # Get the original PDF path from metadata
            pdf_path = doc["metadata"].get("Source-File")
            if not pdf_path:
                continue

            # Get page spans from attributes
            page_spans = doc["attributes"].get("pdf_page_numbers", [])
            if not page_spans:
                continue

            # Pick a random page span
            page_span = random.choice(page_spans)
            if len(page_span) >= 3:
                # Page spans are [start_pos, end_pos, page_num]
                page_num = page_span[2]

                # Extract text for this page
                start_pos, end_pos = page_span[0], page_span[1]
                page_text = doc["text"][start_pos:end_pos].strip()

                # Include the text snippet with the page info
                random_pages.append((pdf_path, page_num, page_text, result_file))

                if len(random_pages) >= count:
                    break

        except Exception as e:
            print(f"Error processing {result_file}: {e}")
            continue

    print(f"Found {len(random_pages)} random pages from Dolma documents")
    return random_pages


def create_presigned_url(s3_client, pdf_path, expiration=3600 * 24 * 7):
    """Create a presigned URL for the given S3 path."""
    try:
        bucket, key = parse_s3_path(pdf_path)
        url = s3_client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expiration)
        return url
    except Exception as e:
        print(f"Error creating presigned URL for {pdf_path}: {e}")
        return None


def create_html_output(random_pages, pdf_s3_client, output_path, workspace_path, db_path, resolution=2048):
    """Create an HTML file with rendered PDF pages."""
    # Get current date and time for the report
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OLMO OCR Samples</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {{
                --primary-color: #2563eb;
                --secondary-color: #4b5563;
                --border-color: #e5e7eb;
                --bg-color: #f9fafb;
                --text-color: #111827;
                --text-light: #6b7280;
                --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            }}
            
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}
            
            body {{
                font-family: 'Inter', sans-serif;
                line-height: 1.6;
                color: var(--text-color);
                background-color: var(--bg-color);
                padding: 2rem;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            
            header {{
                margin-bottom: 2rem;
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 1rem;
            }}
            
            header h1 {{
                color: var(--primary-color);
                font-size: 2rem;
                margin-bottom: 0.5rem;
            }}
            
            header p {{
                color: var(--secondary-color);
                font-size: 1rem;
            }}
            
            .info-bar {{
                background-color: white;
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 2rem;
                box-shadow: var(--card-shadow);
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 1rem;
            }}
            
            .info-item {{
                flex: 1;
                min-width: 200px;
            }}
            
            .info-item h3 {{
                font-size: 0.875rem;
                color: var(--text-light);
                margin-bottom: 0.25rem;
            }}
            
            .info-item p {{
                font-size: 1rem;
                font-weight: 500;
            }}
            
            .page-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 2rem;
            }}
            
            .page-container {{
                background-color: white;
                border-radius: 0.5rem;
                overflow: hidden;
                box-shadow: var(--card-shadow);
            }}
            
            .page-info {{
                padding: 1rem;
                border-bottom: 1px solid var(--border-color);
            }}
            
            .page-info h2 {{
                font-size: 1rem;
                margin-bottom: 0.5rem;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }}
            
            .page-info p {{
                font-size: 0.875rem;
                color: var(--text-light);
            }}
            
            .page-image-wrapper {{
                padding: 1rem;
                display: flex;
                justify-content: center;
                align-items: center;
                background-color: #f3f4f6;
            }}
            
            .page-image {{
                max-width: 100%;
                height: auto;
                border: 1px solid var(--border-color);
            }}
            
            .s3-link {{
                padding: 1rem;
                background-color: #f8fafc;
                border-top: 1px solid var(--border-color);
                font-size: 0.875rem;
                color: var(--secondary-color);
                word-break: break-all;
            }}
            
            .s3-link a {{
                color: var(--primary-color);
                text-decoration: none;
                font-weight: 500;
            }}
            
            .s3-link a:hover {{
                text-decoration: underline;
            }}
            
            .error {{
                color: #dc2626;
                padding: 1rem;
                background-color: #fee2e2;
                border-radius: 0.25rem;
            }}
            
            footer {{
                margin-top: 3rem;
                text-align: center;
                color: var(--text-light);
                font-size: 0.875rem;
                border-top: 1px solid var(--border-color);
                padding-top: 1rem;
            }}
            
            @media (max-width: 768px) {{
                body {{
                    padding: 1rem;
                }}
                
                .page-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>OLMOCR Random Samples</h1>
                <p>A visual survey of randomly selected pages from processed documents</p>
            </header>
            
            <div class="info-bar">
                <div class="info-item">
                    <h3>Generated On</h3>
                    <p>{current_time}</p>
                </div>
                <div class="info-item">
                    <h3>Workspace</h3>
                    <p title="{workspace_path}">{workspace_path}</p>
                </div>
                <div class="info-item">
                    <h3>Sample Size</h3>
                    <p>{len(random_pages)} pages</p>
                </div>
            </div>
            
            <div class="page-grid">
    """

    for i, (pdf_path, page_num, page_text, result_file) in enumerate(tqdm(random_pages, desc="Rendering pages")):
        # Get original URL from PDF hash
        pdf_hash = parse_pdf_hash(pdf_path)
        original_url = get_original_url(pdf_hash, db_path) if pdf_hash else None

        # Create a truncated path for display
        display_path = pdf_path
        if len(display_path) > 60:
            display_path = "..." + display_path[-57:]

        # Generate presigned URL
        presigned_url = create_presigned_url(pdf_s3_client, pdf_path)

        try:
            # Download PDF to temp file
            bucket, key = parse_s3_path(pdf_path)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                pdf_data = pdf_s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
                temp_file.write(pdf_data)
                temp_file_path = temp_file.name

            # Render PDF to base64 webp
            base64_image = render_pdf_to_base64webp(temp_file_path, page_num, resolution)

            # Add to HTML
            html_content += f"""
            <div class="page-container">
                <div class="page-info">
                    <h2 title="{pdf_path}"><a href="{original_url}" target="_blank">{original_url}</a></h2>
                    <p>Page {page_num}</p>
                    <p>{f'<a href="{presigned_url}" target="_blank">View Cached PDF</a>' if presigned_url else pdf_path}</p>
                </div>
                <div class="page-image-wrapper">
                    <img class="page-image" src="data:image/webp;base64,{base64_image}" alt="PDF Page {page_num}" loading="lazy" />
                </div>
            </div>
            """


            # Clean up temp file
            os.unlink(temp_file_path)

        except Exception as e:
            html_content += f"""
            <div class="page-container">
                <div class="page-info">
                    <h2 title="{pdf_path}"><a href="{original_url}" target="_blank">{original_url}</a></h2>
                    <p>Page {page_num}</p>
                    <p>{f'<a href="{presigned_url}" target="_blank">View Cached PDF</a>' if presigned_url else pdf_path}</p>
                </div>
                <div class="error">Error: {str(e)}</div>
            </div>
            """

    html_content += """
            </div>
            <footer>
                <p>Generated by OLMOCR Sampling Tool</p>
            </footer>
        </div>
    </body>
    </html>
    """

    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"Created HTML output at {output_path}")


def generate_sample_set(args, i, s3_client, pdf_s3_client, result_files):
    """Generate a single sample set."""
    output_filename = Path(args.output_dir) / f"dolma_samples_{i+1}.html"

    print(f"\nGenerating sample set {i+1} of {args.repeats}")

    # Get random pages
    random_pages = get_random_pages(s3_client, result_files, args.pages_per_output)

    # Create HTML output
    create_html_output(random_pages, pdf_s3_client, output_filename, args.workspace, args.db_path)

    return output_filename


def main():
    args = parse_args()

    # Set up S3 clients
    s3_client = boto3.client("s3")

    # Set up PDF S3 client with profile if specified
    if args.pdf_profile:
        pdf_session = boto3.Session(profile_name=args.pdf_profile)
        pdf_s3_client = pdf_session.client("s3")
    else:
        pdf_s3_client = s3_client

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # List all result files
    print(f"Listing result files in {args.workspace}/results...")
    result_files = list_result_files(s3_client, args.workspace)
    print(f"Found {len(result_files)} result files")

    # Use ThreadPoolExecutor to parallelize the generation of sample sets
    if args.repeats > 1:
        print(f"Using ThreadPoolExecutor with {min(args.max_workers, args.repeats)} workers")
        with ThreadPoolExecutor(max_workers=min(args.max_workers, args.repeats)) as executor:
            futures = []
            for i in range(args.repeats):
                future = executor.submit(generate_sample_set, args, i, s3_client, pdf_s3_client, result_files)
                futures.append(future)

            # Wait for all futures to complete and collect results
            for future in futures:
                try:
                    output_filename = future.result()
                    print(f"Completed generation of {output_filename}")
                except Exception as e:
                    print(f"Error generating sample set: {e}")
    else:
        # If only one repeat, just run it directly
        generate_sample_set(args, 0, s3_client, pdf_s3_client, result_files)


if __name__ == "__main__":
    main()
