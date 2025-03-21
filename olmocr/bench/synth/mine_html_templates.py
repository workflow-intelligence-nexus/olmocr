import argparse
import concurrent.futures
import os
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pypdf
from anthropic import Anthropic
from tqdm import tqdm

from olmocr.data.renderpdf import render_pdf_to_base64png


def download_s3_pdf(s3_path, local_path):
    """Download a PDF from S3 to a local path."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    result = subprocess.run(["aws", "s3", "cp", s3_path, local_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode == 0


def generate_html_from_image(client, image_base64):
    """Call Claude API to generate HTML from an image."""
    try:
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4000,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}},
                        {
                            "type": "text",
                            "text": "Render this document as clean, semantic HTML. Use appropriate HTML tags for elements like headings, paragraphs, lists, tables, etc. "
                            "Use the <header> and <footer> tags to represent content at the top/bottom which would not normally be part of the main content, such as page numbers, etc. "
                            "Use a placeholder <div> tag with class 'image' which will render as a grey box with black outline to make sure images have their original size, shape, and position on the page. "
                            "Preserve any multi-column layouts exactly as they appear. "
                            "Focus on creating valid, accessible HTML that preserves the appearance and formatting of the original page as closely as possible. ",
                        },
                    ],
                }
            ],
        )

        # Extract HTML from response
        html_content = ""
        for content in response.content:
            if content.type == "text":
                html_content += content.text

        # Extract code blocks from response if HTML is wrapped in them
        if "```html" in html_content:
            start = html_content.find("```html") + 7
            end = html_content.rfind("```")
            if end > start:
                html_content = html_content[start:end].strip()
        elif "```" in html_content:
            start = html_content.find("```") + 3
            end = html_content.rfind("```")
            if end > start:
                html_content = html_content[start:end].strip()

        return html_content
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return None


def extract_page_from_pdf(input_path, output_path, page_num):
    """
    Extract a specific page from a PDF and save it as a new PDF.

    Args:
        input_path: Path to the input PDF
        output_path: Path to save the extracted page
        page_num: The page number to extract (1-indexed, converted to 0-indexed for pypdf)

    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Read the input PDF
        reader = pypdf.PdfReader(input_path)

        # Convert to 0-indexed for pypdf
        zero_idx_page = page_num - 1

        # Check if page number is valid
        if zero_idx_page >= len(reader.pages) or zero_idx_page < 0:
            print(f"Page number {page_num} out of range for {input_path} with {len(reader.pages)} pages")
            return False

        # Create a new PDF with just the selected page
        writer = pypdf.PdfWriter()
        writer.add_page(reader.pages[zero_idx_page])

        # Write the output PDF
        with open(output_path, "wb") as output_file:
            writer.write(output_file)

        return True
    except Exception as e:
        print(f"Error extracting page {page_num} from {input_path}: {str(e)}")
        return False


def process_pdf(pdf_info, args, client):
    """Process a single PDF, render a random page, and create an HTML template."""
    s3_path, index = pdf_info

    # Create a unique folder for each PDF in the temp directory
    pdf_id = f"pdf_{index:05d}"
    temp_pdf_dir = os.path.join(args.temp_dir, pdf_id)
    os.makedirs(temp_pdf_dir, exist_ok=True)

    # Download PDF to local temp directory
    local_pdf_path = os.path.join(temp_pdf_dir, "document.pdf")
    if not download_s3_pdf(s3_path, local_pdf_path):
        print(f"Failed to download PDF from {s3_path}")
        return None

    try:
        # Get page count using pypdf
        reader = pypdf.PdfReader(local_pdf_path)
        num_pages = len(reader.pages)

        if num_pages == 0:
            print(f"PDF has no pages: {s3_path}")
            return None

        # Select a random page
        page_num = random.randint(1, num_pages)

        # Render the page as a base64 PNG
        image_base64 = render_pdf_to_base64png(local_pdf_path, page_num, target_longest_image_dim=2048)

        # Generate HTML from the image
        html_content = generate_html_from_image(client, image_base64)
        if not html_content:
            print(f"Failed to generate HTML for {s3_path}, page {page_num}")
            return None

        # Create output directory
        templates_dir = os.path.join(args.output_dir, "templates")
        os.makedirs(templates_dir, exist_ok=True)

        # Save HTML to output directory
        html_path = os.path.join(templates_dir, f"{pdf_id}_page{page_num}.html")
        with open(html_path, "w") as f:
            f.write(html_content)

        # Extract the page and save as PDF
        pdf_path = os.path.join(templates_dir, f"{pdf_id}_page{page_num}.pdf")
        if not extract_page_from_pdf(local_pdf_path, pdf_path, page_num):
            print(f"Failed to extract page {page_num} from {local_pdf_path}")

        return {"pdf_id": pdf_id, "s3_path": s3_path, "page_number": page_num, "html_path": html_path, "pdf_path": pdf_path}
    except Exception as e:
        print(f"Error processing {s3_path}: {e}")
        return None
    finally:
        # Clean up temp directory for this PDF
        if os.path.exists(temp_pdf_dir):
            subprocess.run(["rm", "-rf", temp_pdf_dir])


def main():
    parser = argparse.ArgumentParser(description="Convert PDFs to HTML templates")
    parser.add_argument("--input_list", required=True, help="Path to a file containing S3 paths to PDFs")
    parser.add_argument("--output_dir", required=True, help="Directory to store extracted pages and tests")
    parser.add_argument("--temp_dir", default="/tmp/mine_tables", help="Directory for temporary files")
    parser.add_argument("--max_tests", type=int, default=100, help="Maximum number of tests to generate")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel threads to use")
    parser.add_argument("--api_key", help="Claude API key (or set ANTHROPIC_API_KEY environment variable)")
    args = parser.parse_args()

    # Ensure output and temp directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    # Get API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: API key not provided. Use --api_key or set ANTHROPIC_API_KEY environment variable.")
        return

    # Initialize Claude client
    client = Anthropic(api_key=api_key)

    # Reservoir sampling implementation
    s3_paths = []
    with open(args.input_list, "r") as f:
        for i, line in enumerate(tqdm(f)):
            line = line.strip()
            if not line:
                continue

            if i < 100000:
                s3_paths.append(line)
            else:
                # Randomly replace elements with decreasing probability
                j = random.randint(0, i)
                if j < 100000:
                    s3_paths[j] = line

    print(f"Found {len(s3_paths)} PDF paths in input list")

    # Shuffle and limit to max_tests
    random.shuffle(s3_paths)
    s3_paths = s3_paths[: args.max_tests]

    # Process PDFs in parallel
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        # Submit all tasks
        futures = {executor.submit(process_pdf, (s3_path, i), args, client): s3_path for i, s3_path in enumerate(s3_paths)}

        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing PDFs"):
            s3_path = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing {s3_path}: {e}")

    print(f"Generated {len(results)} HTML templates")


if __name__ == "__main__":
    main()
