import argparse
import base64
import csv
import datetime
import json
import os
import random
import re
import sqlite3
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import requests
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
        "--prolific_code",
        required=True,
        help="Fixed completion code to use for all outputs",
    )
    parser.add_argument(
        "--prolific_csv",
        default="prolific_codes.csv",
        help="Path to save the file with tinyhost links (one URL per line)",
    )
    parser.add_argument(
        "--read_results",
        help="Path to a CSV file containing previously generated tinyhost links to extract annotations",
    )
    return parser.parse_args()


# Fixed prolific code is now passed in as a command line argument


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
                display: flex;
                flex-direction: row;
                gap: 2rem;
            }}
            
            ul {{
                margin-left: 2em;
            }}

            .container {{
                flex: 2;
                max-width: 750px;
            }}
            
            header {{
                position: sticky;
                top: 2rem;
                flex: 1;
                min-width: 380px;
                max-width: 420px;
                max-height: calc(100vh - 4rem);
                overflow-y: auto;
                padding: 1.5rem;
                background-color: white;
                border-radius: 0.5rem;
                box-shadow: var(--card-shadow);
                align-self: flex-start;
                font-size: small;
            }}

            header h2 {{
                margin-top: 1em;
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
                grid-template-columns: 1fr;
                gap: 2rem;
            }}
            
            .page-container {{
                background-color: white;
                border-radius: 0.5rem;
                overflow: hidden;
                box-shadow: var(--card-shadow);
                transition: all 0.3s ease;
            }}
            
            .page-container.editing {{
                box-shadow: 0 0 0 3px var(--primary-color), var(--card-shadow);
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
                cursor: pointer;
                transition: all 0.2s ease;
            }}
            
            .status-complete:hover {{
                background-color: #d1fae5;
                box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.3);
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
                    flex-direction: column;
                }}
                
                header {{
                    position: static;
                    max-width: 100%;
                    margin-left: 0;
                    margin-bottom: 2rem;
                }}
                
                .container {{
                    max-width: 100%;
                }}
            }}
        </style>
    </head>
    <body>
        <header>
            <h2>Task Instructions</h2>
            <p>Your task is to review {len(random_pages)} document pages and determine whether they contain any <strong>Personally Identifiable Information (PII)</strong>. Carefully but efficiently inspect each page and select the appropriate response. You do not need to read every word - quickly scan the page and look for any obvious PII. The time expected to complete this task is 10-15 minutes.</p>
            
            <h2>How to Annotate</h2>
            <p>The page you are currently annotating will be highlighted with a blue outline and a set of response buttons will be displayed directly below it.</p>
            <br/>
            <p><strong>Yes PII</strong> - Select this if you find any information on the page that qualifies as PII. A text box will appear below - briefly describe the kind of PII you encountered (e.g., full name, social security number, etc.) then press the Enter key.</p>
            <p><strong>No PII</strong> - Select this if the page does not contain any PII.</p>
            <p><strong>I cannot read this</strong> - Select this if you are unable to read the page for any reason (e.g., written in a language other than English, heavily redacted text, etc.)</p>
            <p><strong>Disturbing content</strong> - Select this if the page contains disturbing or graphic content.</p>
            
            <br/>
            <p>You may edit your annotations any time before submitting. To do so, press the green Edit button directly above the page.</p>
            <p>After completing all the document pages on this screen, you will receive a Prolific completion code.</p>
            
            <h2>What Counts as PII?</h2>
            <ul>
                <li><strong>Names</strong>: Full names, first names, last names, nicknames, maiden names, aliases</li>
                <li><strong>Addresses</strong>: Street addresses, postal codes, cities, states, countries</li>
                <li><strong>Contact Information</strong>: Phone numbers, email addresses</li>
                <li><strong>Government IDs</strong>: SSNs, passport numbers, driver's license numbers, tax IDs</li>
                <li><strong>Financial Information</strong>: Credit card numbers, bank account numbers, routing numbers</li>
                <li><strong>Biometric Data</strong>: Fingerprints, retina scans, facial recognition data, voice signatures</li>
                <li><strong>Personal Attributes</strong>: Date of birth, place of birth, gender, race, religion</li>
                <li><strong>Online Identifiers</strong>: IP addresses, login IDs, usernames, passwords, API keys, URLs</li>
                <li><strong>Location Information</strong>: Geolocations, specific coordinates</li>
                <li><strong>Employment Information</strong>: Job titles, workplace names, employment history</li>
                <li><strong>Education Information</strong>: School names, degrees, transcripts</li>
                <li><strong>Medical Information</strong>: Health records, diagnoses</li>
                <li><strong>Company Names</strong>: If they are tied to an individual's identity (e.g., a person's personal business)</li>
            </ul>
            
            <h2>What NOT to Mark as PII</h2>
            <p><strong>Author names, researcher names, citations, or references from published research papers</strong> should NOT be marked as PII. These names are part of the normal publication process and are not considered private or sensitive information for the purposes of this task.
            Only mark information as PII if it relates to private, sensitive, or personal details about an individual outside the context of the publication.</p>
        </header>
        <div class="container">
            
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
                    <h2 title="{pdf_path}">{original_url}</h2>
                    <p>Page {page_num}</p>
                    <p>{f'<a href="{presigned_url}" target="_blank">View Cached PDF</a>' if presigned_url else pdf_path}</p>
                    <p>
                        Status: <span class="annotation-status status-pending" id="status-{i}">Pending</span>
                    </p>
                </div>
                <div class="page-image-wrapper">
                    <img class="page-image" src="data:image/webp;base64,{base64_image}" alt="PDF Page {page_num}" loading="lazy" />
                </div>
                <div class="annotation-interface{active_class}" data-id="page-{i}" data-pdf-path="{pdf_path}">
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
                    <h2 title="{pdf_path}">original_url</h2>
                    <p>Page {page_num}</p>
                    <p>{f'<a href="{presigned_url}" target="_blank">View Cached PDF</a>' if presigned_url else pdf_path}</p>
                    <p>
                        Status: <span class="annotation-status status-pending" id="status-{i}">Pending</span>
                    </p>
                </div>
                <div class="error">Error: {str(e)}</div>
                <div class="annotation-interface{active_class}" data-id="page-{i}" data-pdf-path="{pdf_path}">
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
            
        </div>
        <script>
            // Using externally injected async functions: fetchDatastore() and putDatastore()
            
            // Track annotation progress
            let currentIndex = 0;
            const totalPages = document.querySelectorAll('.page-container').length;
     
            // Update progress bar
            function updateProgressBar() {
                // Check if all annotations are complete
                if (currentIndex >= totalPages) {
                    document.getElementById('completion-message').style.display = 'block';
                }
            }
            
            // Update status indicators
            function updateStatusIndicators() {
                // Reset all status indicators
                document.querySelectorAll('.annotation-status').forEach(function(status) {
                    status.className = 'annotation-status status-pending';
                    status.textContent = 'Pending';
                    // Remove any click handlers
                    status.onclick = null;
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
                        status.textContent = 'Edit ✎';
                        // Add click handler to edit this annotation
                        status.onclick = function() { editAnnotation(i); };
                    }
                }
            }
            
            // Function to enable editing a previously completed annotation
            function editAnnotation(index) {
                // Hide current annotation interface
                document.querySelector(`.annotation-interface[data-id="page-${currentIndex}"]`).classList.remove('active');
                
                // Remove editing class from all containers
                document.querySelectorAll('.page-container').forEach(container => {
                    container.classList.remove('editing');
                });
                
                // Show the selected annotation interface
                document.querySelector(`.annotation-interface[data-id="page-${index}"]`).classList.add('active');
                
                // Add editing class to the container being edited
                const activeContainer = document.querySelector(`.page-container[data-index="${index}"]`);
                if (activeContainer) {
                    activeContainer.classList.add('editing');
                    activeContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
                
                // Update current index
                currentIndex = index;
                updateProgressBar();
                updateStatusIndicators();
            }
            
            // Navigate to the next document
            function goToNextDocument() {
                // Hide current annotation interface
                document.querySelector(`.annotation-interface[data-id="page-${currentIndex}"]`).classList.remove('active');
                
                // Remove editing class from all containers
                document.querySelectorAll('.page-container').forEach(container => {
                    container.classList.remove('editing');
                });
                
                // Move to next document if not at the end
                if (currentIndex < totalPages - 1) {
                    currentIndex++;
                    document.querySelector(`.annotation-interface[data-id="page-${currentIndex}"]`).classList.add('active');
                    
                    // Add editing class to current container
                    const activeContainer = document.querySelector(`.page-container[data-index="${currentIndex}"]`);
                    if (activeContainer) {
                        activeContainer.classList.add('editing');
                        activeContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                    
                    updateProgressBar();
                    updateStatusIndicators();
                }
                else {
                    // This was the last document, mark as complete
                    currentIndex = totalPages;
                    updateProgressBar();
                    updateStatusIndicators();
                    
                    // Show completion message and scroll to it
                    document.getElementById('completion-message').style.display = 'block';
                    document.getElementById('completion-message').scrollIntoView({ behavior: 'smooth', block: 'center' });
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
                const pdfPath = interfaceDiv.getAttribute('data-pdf-path');

                const datastore = await fetchDatastore() || {};
                datastore[id] = {
                    feedbackOption: feedbackOption,
                    piiDescription: piiDescription,
                    pdfPath: pdfPath
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
                
                // Add editing class to the first container by default
                const firstContainer = document.querySelector(`.page-container[data-index="0"]`);
                if (firstContainer) {
                    firstContainer.classList.add('editing');
                }
                
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
                if (lastAnnotatedIndex >= 0) {
                    document.querySelector(`.annotation-interface.active`).classList.remove('active');
                    
                    // Check if all pages are annotated
                    if (lastAnnotatedIndex === totalPages - 1) {
                        // All pages are annotated, set currentIndex to totalPages to trigger completion
                        currentIndex = totalPages;
                        
                        // Show completion message and scroll to it
                        document.getElementById('completion-message').style.display = 'block';
                        document.getElementById('completion-message').scrollIntoView({ behavior: 'smooth', block: 'center' });
                    } else {
                        // Go to the next unannotated page
                        currentIndex = lastAnnotatedIndex + 1;
                        document.querySelector(`.annotation-interface[data-id="page-${currentIndex}"]`).classList.add('active');
                        
                        // Add editing class and scroll to the active annotation
                        const activeContainer = document.querySelector(`.page-container[data-index="${currentIndex}"]`);
                        if (activeContainer) {
                            // Remove editing class from all containers first
                            document.querySelectorAll('.page-container').forEach(container => {
                                container.classList.remove('editing');
                            });
                            // Add editing class to current container
                            activeContainer.classList.add('editing');
                            activeContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        }
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

    # Use the fixed prolific code from command line arguments
    prolific_code = args.prolific_code

    # Create HTML output with the Prolific code
    create_html_output(random_pages, pdf_s3_client, output_filename, args.workspace, args.db_path, prolific_code)

    return output_filename


def extract_datastore_url(html_content: str) -> Optional[str]:
    """Extract the presigned datastore URL from HTML content."""
    match = re.search(r'const\s+presignedGetUrl\s*=\s*"([^"]+)"', html_content)
    if match:
        return match.group(1)
    return None


def fetch_annotations(tinyhost_link: str) -> Tuple[Dict[str, Any], str]:
    """Fetch and parse annotations from a tinyhost link."""
    # Request the HTML content
    print(f"Fetching annotations from {tinyhost_link}")
    response = requests.get(tinyhost_link)
    response.raise_for_status()
    html_content = response.text

    # Extract the datastore URL
    datastore_url = extract_datastore_url(html_content)
    if not datastore_url:
        print(f"Could not find datastore URL in {tinyhost_link}")
        return {}, tinyhost_link

    # Fetch the datastore content
    print(f"Found datastore URL: {datastore_url}")
    try:
        datastore_response = requests.get(datastore_url)
        datastore_response.raise_for_status()
        annotations = datastore_response.json()
        return annotations, tinyhost_link
    except Exception as e:
        print(f"Error fetching datastore from {datastore_url}: {e}")
        return {}, tinyhost_link


def process_annotations(annotations_by_link: List[Tuple[Dict[str, Any], str]]) -> Dict[str, List[Dict[str, Any]]]:
    """Process and categorize annotations by feedback type."""
    results = {
        "yes_pii": [],
        "no_pii": [],
        "cannot_read": [],
        "disturbing": [],
        "no_annotation": [],
    }

    # Process each annotation
    for annotations, link in annotations_by_link:
        for page_id, annotation in annotations.items():
            if not annotation or "feedbackOption" not in annotation:
                results["no_annotation"].append(
                    {"page_id": page_id, "link": link, "pdf_path": annotation.get("pdfPath", "Unknown") if annotation else "Unknown"}
                )
                continue

            category = annotation["feedbackOption"]
            result_item = {
                "page_id": page_id,
                "link": link,
                "pdf_path": annotation.get("pdfPath", "Unknown"),
                "description": annotation.get("piiDescription", ""),
            }

            if category == "yes-pii":
                results["yes_pii"].append(result_item)
            elif category == "no-pii":
                results["no_pii"].append(result_item)
            elif category == "cannot-read":
                results["cannot_read"].append(result_item)
            elif category == "disturbing":
                results["disturbing"].append(result_item)
            else:
                results["no_annotation"].append(result_item)

    return results


def print_annotation_report(annotation_results: Dict[str, List[Dict[str, Any]]]):
    """Print a summary report of annotations."""
    total_pages = sum(len(items) for items in annotation_results.values())

    print("\n" + "=" * 80)
    print(f"ANNOTATION REPORT - Total Pages: {total_pages}")
    print("=" * 80)

    # Print summary statistics
    print("\nSummary:")
    print(f"  Pages with PII: {len(annotation_results['yes_pii'])} ({len(annotation_results['yes_pii'])/total_pages*100:.1f}%)")
    print(f"  Pages without PII: {len(annotation_results['no_pii'])} ({len(annotation_results['no_pii'])/total_pages*100:.1f}%)")
    print(f"  Unreadable pages: {len(annotation_results['cannot_read'])} ({len(annotation_results['cannot_read'])/total_pages*100:.1f}%)")
    print(f"  Pages with disturbing content: {len(annotation_results['disturbing'])} ({len(annotation_results['disturbing'])/total_pages*100:.1f}%)")
    print(f"  Pages without annotation: {len(annotation_results['no_annotation'])} ({len(annotation_results['no_annotation'])/total_pages*100:.1f}%)")

    # Print detailed report for pages with PII
    if annotation_results["yes_pii"]:
        print("\nDetailed Report - Pages with PII:")
        print("-" * 80)
        for i, item in enumerate(annotation_results["yes_pii"], 1):
            print(f"{i}. PDF: {item['pdf_path']}")
            print(f"   Page ID: {item['page_id']}")
            print(f"   Link: {item['link']}#{item['page_id']}")
            print(f"   Description: {item['description']}")
            print("-" * 80)

    print("\nReport complete.")


def read_and_process_results(args):
    """Read and process results from a previously generated CSV file."""
    try:
        # Read the CSV file
        links = []
        with open(args.read_results, "r") as f:
            for line in f:
                if line.strip():
                    links.append(line.strip())

        if not links:
            print(f"No tinyhost links found in {args.read_results}")
            return

        print(f"Found {len(links)} tinyhost links in {args.read_results}")

        # Fetch and process annotations
        annotations_by_link = []
        for link in tqdm(links, desc="Fetching annotations"):
            try:
                annotations, link_url = fetch_annotations(link)
                annotations_by_link.append((annotations, link_url))
            except Exception as e:
                print(f"Error processing {link}: {e}")

        # Process and categorize annotations
        annotation_results = process_annotations(annotations_by_link)

        # Print report
        print_annotation_report(annotation_results)

        # Save detailed report to file
        output_file = Path(args.output_dir) / "annotation_report.csv"
        print(f"\nSaving detailed report to {output_file}")

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Category", "PDF Path", "Page ID", "Link", "Description"])

            for category, items in annotation_results.items():
                for item in items:
                    writer.writerow([category, item["pdf_path"], item["page_id"], f"{item['link']}#{item['page_id']}", item.get("description", "")])

        print(f"Report saved to {output_file}")

    except Exception as e:
        print(f"Error processing results: {e}")


def main():
    args = parse_args()

    # Check if we're reading results from a previous run
    if args.read_results:
        read_and_process_results(args)
        return

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
                    output_files.append(output_filename)
                    print(f"Completed generation of {output_filename}")
                except Exception as e:
                    print(f"Error generating sample set: {e}")
    else:
        # If only one repeat, just run it directly
        output_filename = generate_sample_set(args, 0, s3_client, pdf_s3_client, result_files)
        output_files.append(output_filename)

    # Now upload each resulting file into tinyhost
    print("Generated all files, uploading tinyhost links now")
    links = []
    for output_filename in output_files:
        link = tinyhost.tinyhost([str(output_filename)])[0]
        links.append(link)
        print(link)

    # Create CSV file with just the tinyhost links, one per line
    csv_path = args.prolific_csv
    print(f"Writing tinyhost links to {csv_path}")
    with open(csv_path, "w", newline="") as csvfile:
        for link in links:
            csvfile.write(f"{link}\n")

    print(f"Tinyhost links written to {csv_path}")


if __name__ == "__main__":
    main()
