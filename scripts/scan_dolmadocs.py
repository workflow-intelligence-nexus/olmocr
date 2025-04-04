import argparse
import base64
import csv
import datetime
import json
import os
import random
import re
import sqlite3
import string
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import boto3
import tinyhost
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
    parser.add_argument(
        "--prolific_csv",
        default="prolific_codes.csv",
        help="Path to save the CSV file with Prolific codes",
    )
    return parser.parse_args()


def generate_prolific_code(length=8):
    """Generate a random code for Prolific."""
    characters = string.ascii_uppercase + string.digits
    return "".join(random.choice(characters) for _ in range(length))


def obfuscate_code(code):
    """Gently obfuscate the Prolific code so it's not immediately visible in source."""
    # Convert to base64 and reverse
    encoded = base64.b64encode(code.encode()).decode()
    return encoded[::-1]


def deobfuscate_code(obfuscated_code):
    """Deobfuscate the code - this will be done in JavaScript."""
    # Reverse and decode from base64
    reversed_encoded = obfuscated_code[::-1]
    try:
        return base64.b64decode(reversed_encoded).decode()
    except:
        return "ERROR_DECODING"


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


def create_html_output(random_pages, pdf_s3_client, output_path, workspace_path, db_path, prolific_code, resolution=2048):
    """Create an HTML file with rendered PDF pages."""
    # Obfuscate the provided Prolific code
    obfuscated_code = obfuscate_code(prolific_code)

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
                --success-color: #10b981;
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
                font-size: 0.6rem;
                color: var(--text-light);
                margin-bottom: 0.25rem;
            }}
            
            .info-item p {{
                font-size: 0.6rem;
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
            
            /* Annotation elements */
            .annotation-interface {{
                display: none; /* Hide annotation interface by default */
                margin-top: 1rem;
                padding: 0.5rem;
                border-top: 1px solid var(--border-color);
                border-radius: 0.25rem;
                background-color: #f8fafc;
            }}
            
            .annotation-interface.active {{
                display: block; /* Show only the active annotation interface */
            }}
            
            /* Button group styling for connected buttons */
            .btn-group {{
                display: inline-flex;
                margin-bottom: 0.5rem;
            }}
            
            .btn-group .toggle-button {{
                padding: 0.5rem 1rem;
                border: 1px solid var(--border-color);
                background-color: #f8fafc;
                cursor: pointer;
                margin: 0;
                /* Remove individual border radius so we can set unified ones */
                border-radius: 0;
            }}
            
            .btn-group .toggle-button:first-child {{
                border-right: none;
                border-top-left-radius: 0.25rem;
                border-bottom-left-radius: 0.25rem;
            }}
            
            .btn-group .toggle-button:last-child {{
                border-top-right-radius: 0.25rem;
                border-bottom-right-radius: 0.25rem;
            }}
            
            .btn-group .toggle-button:not(:first-child):not(:last-child) {{
                border-right: none;
            }}
            
            .toggle-button.active {{
                background-color: var(--primary-color);
                color: white;
            }}
            
            .annotation-interface textarea {{
                display: none; /* Hide textarea by default */
                width: 100%;
                margin-top: 0.5rem;
                padding: 0.5rem;
                font-size: 0.875rem;
                border: 1px solid var(--border-color);
                border-radius: 0.25rem;
            }}
            
            .annotation-status {{
                display: inline-block;
                margin-left: 1rem;
                padding: 0.25rem 0.5rem;
                border-radius: 0.25rem;
                font-size: 0.75rem;
                font-weight: 600;
            }}
            
            .status-complete {{
                background-color: #ecfdf5;
                color: var(--success-color);
            }}
            
            .status-pending {{
                background-color: #fff7ed;
                color: #ea580c;
            }}
            
            .status-current {{
                background-color: #eff6ff;
                color: var(--primary-color);
                animation: pulse 2s infinite;
            }}
            
            @keyframes pulse {{
                0% {{ opacity: 0.6; }}
                50% {{ opacity: 1; }}
                100% {{ opacity: 0.6; }}
            }}
            
            .error {{
                color: #dc2626;
                padding: 1rem;
                background-color: #fee2e2;
                border-radius: 0.25rem;
            }}
            
            .annotation-progress {{
                position: fixed;
                top: 20px;
                left: 50%;
                transform: translateX(-50%);
                background-color: white;
                padding: 1rem;
                border-radius: 0.5rem;
                box-shadow: var(--card-shadow);
                z-index: 100;
                display: flex;
                align-items: center;
                gap: 1rem;
                min-width: 300px;
            }}
            
            .progress-bar {{
                flex-grow: 1;
                height: 8px;
                background-color: var(--border-color);
                border-radius: 4px;
                overflow: hidden;
            }}
            
            .progress-fill {{
                height: 100%;
                background-color: var(--primary-color);
                width: 0%;
                transition: width 0.3s ease;
            }}
            
            .progress-text {{
                font-size: 0.875rem;
                color: var(--text-light);
                white-space: nowrap;
            }}
            
            .completion-message {{
                display: none;
                margin: 2rem auto;
                padding: 1.5rem;
                background-color: #ecfdf5;
                border: 1px solid #A7F3D0;
                border-radius: 0.5rem;
                text-align: center;
                color: var(--success-color);
                font-weight: 600;
                max-width: 500px;
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
                <p>
                <strong>Instructions: </strong>Please review each document below and mark if it contains PII (Personally identifiable information). If you cannot read it (ex. the document is not in English, or is otherwise unreadable), mark it as such. 
                If the document contains disturbing or graphic content, please mark that. Finally, if there is PII, type in a brief description and press Enter. Once you mark all 30 documents, the completetion code will 
                be presented.
                </p>

                <div style="display: flex; font-family: Arial, sans-serif; font-size: 14px; max-width: 1000px; margin: 0 auto;">
                <div style="flex: 1; padding: 15px; background-color: #f5f7f9; border-radius: 8px; margin-right: 10px;">
                    <h3 style="color: #2c3e50; margin-top: 0; border-bottom: 1px solid #ddd; padding-bottom: 8px;">PII Direct Identifiers</h3>
                    <p style="font-style: italic; color: #555; margin-bottom: 15px;">Information that identifies a data subject without further context</p>
                    <ul style="padding-left: 20px; line-height: 1.5; margin-top: 0;">
                    <li>Names: Full names, first names, last names, nicknames, maiden names, birth names, aliases</li>
                    <li>Addresses: Street addresses, postal codes, city, state, country</li>
                    <li>Contact Information: Phone numbers, email addresses</li>
                    <li>Government IDs: Social Security Numbers (SSNs), passport numbers, driver's license numbers, tax identification numbers</li>
                    <li>Financial Information: Credit card numbers, bank account numbers, routing numbers</li>
                    <li>Biometric Data: Fingerprints, retina scans, voice signatures, facial recognition data</li>
                    <li>Date of Birth of data subject</li>
                    <li>Place of Birth of data subject</li>
                    <li>Gender of data subject</li>
                    <li>Race of data subject</li>
                    <li>Religion of data subject</li>
                    </ul>
                </div>
                
                <div style="flex: 1; padding: 15px; background-color: #f5f7f9; border-radius: 8px; margin-left: 10px;">
                    <h3 style="color: #2c3e50; margin-top: 0; border-bottom: 1px solid #ddd; padding-bottom: 8px;">PII Indirect Identifiers</h3>
                    <p style="font-style: italic; color: #555; margin-bottom: 15px;">Information that can be used to identify a data subject in context or in combination with other information</p>
                    <ul style="padding-left: 20px; line-height: 1.5; margin-top: 0;">
                    <li>IP Addresses</li>
                    <li>Login IDs</li>
                    <li>Geolocations</li>
                    <li>Employment Information</li>
                    <li>Education Information</li>
                    <li>Medical Information</li>
                    <li>Usernames</li>
                    <li>Passwords</li>
                    <li>Keys</li>
                    <li>URLs</li>
                    <li>Company Names</li>
                    </ul>
                </div>
                </div>
    
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

            # Add CSS class for the first annotation interface to be active by default
            active_class = " active" if i == 0 else ""

            # Add to HTML with the annotation interface
            html_content += f"""
            <div class="page-container" data-index="{i}">
                <div class="page-info">
                    <h2 title="{pdf_path}"><a href="{original_url}" target="_blank">{original_url}</a></h2>
                    <p>Page {page_num}</p>
                    <p>{f'<a href="{presigned_url}" target="_blank">View Cached PDF</a>' if presigned_url else pdf_path}</p>
                    <p>
                        Status: <span class="annotation-status status-pending" id="status-{i}">Pending</span>
                    </p>
                </div>
                <div class="page-image-wrapper">
                    <img class="page-image" src="data:image/webp;base64,{base64_image}" alt="PDF Page {page_num}" loading="lazy" />
                </div>
                <div class="annotation-interface{active_class}" data-id="page-{i}">
                    <span class="btn-group">
                        <button type="button" class="toggle-button feedback-option" data-value="yes-pii" onclick="toggleFeedbackOption(this)">Yes PII</button>
                        <button type="button" class="toggle-button feedback-option" data-value="no-pii" onclick="toggleFeedbackOption(this)">No PII</button>
                        <button type="button" class="toggle-button feedback-option" data-value="cannot-read" onclick="toggleFeedbackOption(this)">I cannot read this</button>
                        <button type="button" class="toggle-button feedback-option" data-value="disturbing" onclick="toggleFeedbackOption(this)">Disturbing content</button>
                    </span>
                    <textarea placeholder="Describe any private PII in the document" onchange="saveFeedback(this)" onkeydown="handleTextareaKeydown(event, this)"></textarea>
                </div>
            </div>
            """

            # Clean up temp file
            os.unlink(temp_file_path)

        except Exception as e:
            # Add CSS class for the first annotation interface to be active by default
            active_class = " active" if i == 0 else ""

            html_content += f"""
            <div class="page-container" data-index="{i}">
                <div class="page-info">
                    <h2 title="{pdf_path}"><a href="{original_url}" target="_blank">{original_url}</a></h2>
                    <p>Page {page_num}</p>
                    <p>{f'<a href="{presigned_url}" target="_blank">View Cached PDF</a>' if presigned_url else pdf_path}</p>
                    <p>
                        Status: <span class="annotation-status status-pending" id="status-{i}">Pending</span>
                    </p>
                </div>
                <div class="error">Error: {str(e)}</div>
                <div class="annotation-interface{active_class}" data-id="page-{i}">
                    <span class="btn-group">
                        <button type="button" class="toggle-button feedback-option" data-value="yes-pii" onclick="toggleFeedbackOption(this)">Yes PII</button>
                        <button type="button" class="toggle-button feedback-option" data-value="no-pii" onclick="toggleFeedbackOption(this)">No PII</button>
                        <button type="button" class="toggle-button feedback-option" data-value="cannot-read" onclick="toggleFeedbackOption(this)">I cannot read this</button>
                        <button type="button" class="toggle-button feedback-option" data-value="disturbing" onclick="toggleFeedbackOption(this)">Disturbing content</button>
                    </span>
                    <textarea placeholder="Describe any private PII in the document" onchange="saveFeedback(this)" onkeydown="handleTextareaKeydown(event, this)"></textarea>
                </div>
            </div>
            """

    html_content += (
        """
            </div>
            
            <div class="completion-message" id="completion-message">
                Thank you! All annotations are complete.<br>
                Your Prolific completion code is: <strong id="prolific-code">Loading...</strong>
            </div>
            <!-- Store the obfuscated code in a hidden element -->
            <div id="obfuscated-code" style="display:none;">"""
        + obfuscated_code
        + """</div>
            
            <div class="annotation-progress" id="progress-bar">
                <div class="progress-text">
                    Annotation Progress: <span id="current-page">1</span>/<span id="total-pages"></span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill"></div>
                </div>
            </div>
        </div>
        <script>
            // Using externally injected async functions: fetchDatastore() and putDatastore()
            
            // Track annotation progress
            let currentIndex = 0;
            const totalPages = document.querySelectorAll('.page-container').length;
            document.getElementById('total-pages').textContent = totalPages;
            
            // Update progress bar
            function updateProgressBar() {
                const progressPercent = ((currentIndex + 1) / totalPages) * 100;
                document.getElementById('progress-fill').style.width = progressPercent + '%';
                document.getElementById('current-page').textContent = currentIndex + 1;
                
                // Check if all annotations are complete
                if (currentIndex >= totalPages - 1) {
                    document.getElementById('progress-bar').style.display = 'none';
                    document.getElementById('completion-message').style.display = 'block';
                }
            }
            
            // Update status indicators
            function updateStatusIndicators() {
                // Reset all status indicators
                document.querySelectorAll('.annotation-status').forEach(function(status) {
                    status.className = 'annotation-status status-pending';
                    status.textContent = 'Pending';
                });
                
                // Set current item status
                const currentStatus = document.getElementById(`status-${currentIndex}`);
                if (currentStatus) {
                    currentStatus.className = 'annotation-status status-current';
                    currentStatus.textContent = 'Current';
                }
                
                // Update completed statuses
                for (let i = 0; i < currentIndex; i++) {
                    const status = document.getElementById(`status-${i}`);
                    if (status) {
                        status.className = 'annotation-status status-complete';
                        status.textContent = 'Complete';
                    }
                }
            }
            
            // Navigate to the next document
            function goToNextDocument() {
                // Hide current annotation interface
                document.querySelector(`.annotation-interface[data-id="page-${currentIndex}"]`).classList.remove('active');
                
                // Move to next document if not at the end
                if (currentIndex < totalPages - 1) {
                    currentIndex++;
                    document.querySelector(`.annotation-interface[data-id="page-${currentIndex}"]`).classList.add('active');
                    updateProgressBar();
                    updateStatusIndicators();
                    
                    // Scroll to the new active annotation
                    const activeContainer = document.querySelector(`.page-container[data-index="${currentIndex}"]`);
                    if (activeContainer) {
                        activeContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                }
            }
            
            // Handle text area keydown for Enter key
            function handleTextareaKeydown(event, textarea) {
                // If Enter key is pressed and not with Shift key, move to next document
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    saveFeedback(textarea);
                    goToNextDocument();
                }
            }
            
            async function saveFeedback(source) {
                const interfaceDiv = source.closest('.annotation-interface');
                const id = interfaceDiv.getAttribute('data-id');
                // Get the selected feedback option value
                const activeButton = interfaceDiv.querySelector('button.feedback-option.active');
                const feedbackOption = activeButton ? activeButton.getAttribute('data-value') : null;
                const piiDescription = interfaceDiv.querySelector('textarea').value;

                const datastore = await fetchDatastore() || {};
                datastore[id] = {
                    feedbackOption: feedbackOption,
                    piiDescription: piiDescription
                };

                await putDatastore(datastore);
            }

            function toggleFeedbackOption(btn) {
                const interfaceDiv = btn.closest('.annotation-interface');
                // Remove active class from all feedback option buttons in this group
                interfaceDiv.querySelectorAll('button.feedback-option').forEach(function(b) {
                    b.classList.remove('active');
                });
                // Toggle on the clicked button
                btn.classList.add('active');
                saveFeedback(interfaceDiv);
                
                // Show or hide textarea based on selected option
                const textarea = interfaceDiv.querySelector('textarea');
                const feedbackOption = btn.getAttribute('data-value');
                
                if (feedbackOption === 'yes-pii') {
                    // Only show textarea if "Yes PII" is selected
                    textarea.style.display = 'block';
                    textarea.focus();
                } else {
                    // If other options selected, hide textarea and go to next
                    textarea.style.display = 'none';
                    goToNextDocument();
                }
            }

            // Function to deobfuscate the Prolific code
            function deobfuscateCode(obfuscatedCode) {
                // Reverse the string
                const reversed = obfuscatedCode.split('').reverse().join('');
                // Decode from base64
                try {
                    return atob(reversed);
                } catch (e) {
                    return "ERROR_DECODING";
                }
            }
            
            document.addEventListener("DOMContentLoaded", async function() {
                const datastore = await fetchDatastore() || {};
                updateProgressBar();
                updateStatusIndicators();
                
                // Get and deobfuscate the Prolific code
                const obfuscatedCode = document.getElementById('obfuscated-code').textContent;
                const prolificCode = deobfuscateCode(obfuscatedCode);
                document.getElementById('prolific-code').textContent = prolificCode;
                
                document.querySelectorAll('.annotation-interface').forEach(function(interfaceDiv) {
                    const id = interfaceDiv.getAttribute('data-id');
                    if (datastore[id]) {
                        const data = datastore[id];
                        // Set active state for feedback option buttons
                        interfaceDiv.querySelectorAll('button.feedback-option').forEach(function(btn) {
                            if (btn.getAttribute('data-value') === data.feedbackOption) {
                                btn.classList.add('active');
                                
                                // Show textarea if "Yes PII" is selected
                                if (btn.getAttribute('data-value') === 'yes-pii') {
                                    interfaceDiv.querySelector('textarea').style.display = 'block';
                                }
                            } else {
                                btn.classList.remove('active');
                            }
                        });
                        // Set the textarea value
                        interfaceDiv.querySelector('textarea').value = data.piiDescription;
                    }
                });
                
                // If we have stored data, restore the current position
                let lastAnnotatedIndex = -1;
                for (let i = 0; i < totalPages; i++) {
                    const pageId = `page-${i}`;
                    if (datastore[pageId] && datastore[pageId].feedbackOption) {
                        lastAnnotatedIndex = i;
                    }
                }
                
                // If we have annotated pages, go to the first unannotated page
                if (lastAnnotatedIndex >= 0 && lastAnnotatedIndex < totalPages - 1) {
                    document.querySelector(`.annotation-interface.active`).classList.remove('active');
                    currentIndex = lastAnnotatedIndex + 1;
                    document.querySelector(`.annotation-interface[data-id="page-${currentIndex}"]`).classList.add('active');
                    
                    // Scroll to the active annotation
                    const activeContainer = document.querySelector(`.page-container[data-index="${currentIndex}"]`);
                    if (activeContainer) {
                        activeContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                    
                    updateProgressBar();
                    updateStatusIndicators();
                }
            });
        </script>
    </body>
    </html>
    """
    )

    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"Created HTML output at {output_path}")


def generate_sample_set(args, i, s3_client, pdf_s3_client, result_files):
    """Generate a single sample set."""
    output_filename = Path(args.output_dir) / f"dolma_samples_{i+1}.html"

    print(f"\nGenerating sample set {i+1} of {args.repeats}")

    # Get random pages
    random_pages = get_random_pages(s3_client, result_files, args.pages_per_output)

    # Generate a unique Prolific code for this sample set
    prolific_code = generate_prolific_code()

    # Create HTML output with the Prolific code
    create_html_output(random_pages, pdf_s3_client, output_filename, args.workspace, args.db_path, prolific_code)

    return output_filename, prolific_code


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
    output_files = []
    prolific_codes = []

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
                    output_filename, prolific_code = future.result()
                    output_files.append(output_filename)
                    prolific_codes.append(prolific_code)
                    print(f"Completed generation of {output_filename} with code {prolific_code}")
                except Exception as e:
                    print(f"Error generating sample set: {e}")
    else:
        # If only one repeat, just run it directly
        output_filename, prolific_code = generate_sample_set(args, 0, s3_client, pdf_s3_client, result_files)
        output_files.append(output_filename)
        prolific_codes.append(prolific_code)

    # Now upload each resulting file into tinyhost
    print("Generated all files, uploading tinyhost links now")
    links = []
    for output_filename in output_files:
        link = tinyhost.tinyhost([str(output_filename)])
        links.append(link[0])
        print(link)

    # Create CSV file with tinyhost links and Prolific codes
    csv_path = args.prolific_csv
    print(f"Writing Prolific codes to {csv_path}")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tinyhost_link", "code"])
        for link, code in zip(links, prolific_codes):
            writer.writerow([link, code])

    print(f"Prolific codes written to {csv_path}")


if __name__ == "__main__":
    main()
