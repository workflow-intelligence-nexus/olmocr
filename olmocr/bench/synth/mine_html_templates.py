import argparse
import asyncio
import concurrent.futures
import json
import os
import random
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import pypdf
from anthropic import Anthropic
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from syntok.segmenter import process
from tqdm import tqdm

from olmocr.bench.tests import (
    TestType,
)
from olmocr.data.renderpdf import (
    get_png_dimensions_from_base64,
    render_pdf_to_base64png,
)


def download_s3_pdf(s3_path, local_path):
    """Download a PDF from S3 to a local path."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    result = subprocess.run(["aws", "s3", "cp", s3_path, local_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode == 0


def generate_html_from_image(client, image_base64):
    """Call Claude API to generate HTML from an image."""
    png_width, png_height = get_png_dimensions_from_base64(image_base64)

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
                            "If the document has a multi-column layout, you MUST have the same number of columns in your version. "
                            "Focus on creating valid, accessible HTML that preserves the appearance and formatting of the original page as closely as possible. "
                            f"The webpage will be viewed with a fixed viewport size of {png_width // 2} pixels wide by {png_height // 2} pixels tall. "
                            "Before you start, output a basic analysis of the layout and a plan before enclosing your final html in a ```html block.",
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


async def render_pdf_with_playwright(html_content, output_pdf_path, png_width, png_height):
    """
    Render HTML content using Playwright and save it as PDF.
    Try different scale factors if needed to ensure the output is exactly one page.

    Args:
        html_content: HTML content to render
        output_pdf_path: Path to save the rendered PDF
        png_width: Width of the viewport
        png_height: Height of the viewport

    Returns:
        bool: True if rendering was successful with exactly one page, False otherwise
    """
    scale_factors = [1.0, 0.9, 0.8, 0.7]  # Try these scale factors in order

    for scale in scale_factors:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page(viewport={"width": int(png_width // 2 * scale), "height": int(png_height // 2 * scale)})

                # Set the HTML content
                await page.set_content(html_content)

                # Save as PDF with formatting options
                await page.pdf(
                    path=output_pdf_path,
                    scale=scale,
                    print_background=True,
                )

                await browser.close()

                # Check if the output PDF has exactly one page
                try:
                    reader = pypdf.PdfReader(output_pdf_path)
                    if len(reader.pages) == 1:
                        print(f"Successfully rendered as a single page PDF with scale factor {scale}")
                        return True
                    else:
                        print(f"PDF has {len(reader.pages)} pages with scale factor {scale}, trying a smaller scale...")
                        # Continue to the next scale factor
                except Exception as pdf_check_error:
                    print(f"Error checking PDF page count: {pdf_check_error}")
                    return False

        except Exception as e:
            print(f"Error rendering PDF with Playwright at scale {scale}: {str(e)}")
            # Try the next scale factor

    print("Failed to render PDF as a single page with any scale factor")
    return False


def generate_tests_from_html(html_content: str, pdf_id: str, page_num: int) -> List[Dict]:
    """
    Generate tests from HTML content parsed from the PDF.

    Args:
        html_content: The HTML content of the page
        pdf_id: The unique identifier for the PDF
        page_num: The page number

    Returns:
        A list of test dictionaries that can be saved as JSONL
    """
    tests = []
    pdf_filename = f"{pdf_id}_page{page_num}.pdf"
    soup = BeautifulSoup(html_content, "html.parser")

    # Step 1: Process headers, footers, and page numbers for TextAbsenceTests
    headers = soup.find_all("header")
    footers = soup.find_all("footer")
    page_numbers = soup.find_all("div", class_="page-number")

    # Function to create absence tests from text elements
    def create_absence_tests_from_elements(parent_element, element_type):
        # Find all text-containing elements within the parent
        text_elements = []

        # First get direct text nodes within spans, divs, p, and heading tags
        for tag in parent_element.find_all(["span", "div", "p", "h1", "h2", "h3", "h4", "h5", "h6"]):
            text = tag.get_text().strip()
            if text:
                text_elements.append(text)

        # If no elements found, use the parent's text as a fallback
        if not text_elements:
            parent_text = parent_element.get_text().strip()
            if parent_text:
                text_elements.append(parent_text)

        # Create tests for each text element
        for text in text_elements:
            if len(text) > 3:  # Only create tests for meaningful text
                tests.append(
                    {
                        "pdf": pdf_filename,
                        "page": page_num,
                        "id": f"{pdf_id}_{element_type}_{uuid.uuid4().hex[:8]}",
                        "type": TestType.ABSENT.value,
                        "text": text,
                        "max_diffs": 5,
                    }
                )

    # Create TextAbsenceTests for headers
    for header in headers:
        create_absence_tests_from_elements(header, "header")

    # Create TextAbsenceTests for footers
    for footer in footers:
        create_absence_tests_from_elements(footer, "footer")

    # Create TextAbsenceTests for page numbers
    for page_number in page_numbers:
        page_number_text = page_number.get_text().strip()
        if page_number_text:
            tests.append(
                {
                    "pdf": pdf_filename,
                    "page": page_num,
                    "id": f"{pdf_id}_page_number_{uuid.uuid4().hex[:8]}",
                    "type": TestType.ABSENT.value,
                    "text": page_number_text,
                    "max_diffs": 5,
                }
            )

    # Step 2: Generate tests from tables
    tables = soup.find_all("table")
    for table_idx, table in enumerate(tables):
        # Get all cells in the table
        cells = table.find_all(["td", "th"])

        # Skip empty tables or tables with very few cells
        if len(cells) < 4:
            continue

        # Generate tests for some randomly selected cells
        sampled_cells = random.sample(cells, min(3, len(cells)))

        for cell in sampled_cells:
            cell_text = cell.get_text().strip()
            if not cell_text or len(cell_text) < 3:
                continue

            # Find position of this cell in the table
            row = cell.find_parent("tr")
            rows = table.find_all("tr")
            row_idx = rows.index(row)

            # Find cells in this row
            row_cells = row.find_all(["td", "th"])
            col_idx = row_cells.index(cell)

            # Create a TableTest with relevant relationships
            test_data = {
                "pdf": pdf_filename,
                "page": page_num,
                "id": f"{pdf_id}_table{table_idx}_{uuid.uuid4().hex[:8]}",
                "type": TestType.TABLE.value,
                "cell": cell_text,
                "max_diffs": 5,
            }

            # Check cell up
            if row_idx > 0:
                prev_row = rows[row_idx - 1]
                prev_row_cells = prev_row.find_all(["td", "th"])
                if col_idx < len(prev_row_cells):
                    up_text = prev_row_cells[col_idx].get_text().strip()
                    if up_text:
                        test_data["up"] = up_text

            # Check cell down
            if row_idx < len(rows) - 1:
                next_row = rows[row_idx + 1]
                next_row_cells = next_row.find_all(["td", "th"])
                if col_idx < len(next_row_cells):
                    down_text = next_row_cells[col_idx].get_text().strip()
                    if down_text:
                        test_data["down"] = down_text

            # Check cell left
            if col_idx > 0:
                left_text = row_cells[col_idx - 1].get_text().strip()
                if left_text:
                    test_data["left"] = left_text

            # Check cell right
            if col_idx < len(row_cells) - 1:
                right_text = row_cells[col_idx + 1].get_text().strip()
                if right_text:
                    test_data["right"] = right_text

            # Check top heading (first row in the table or a header row)
            if row_idx > 0:
                header_row = rows[0]
                header_cells = header_row.find_all(["td", "th"])
                if col_idx < len(header_cells):
                    top_heading = header_cells[col_idx].get_text().strip()
                    if top_heading:
                        test_data["top_heading"] = top_heading

            # Check left heading (first column in the table)
            if col_idx > 0:
                left_heading = row_cells[0].get_text().strip()
                if left_heading:
                    test_data["left_heading"] = left_heading

            # Only add the test if we have at least one relation
            if len(test_data) > 6:  # 6 is the number of required fields
                tests.append(test_data)

    # Step 3: Generate TextPresenceTests for main body content
    # Make a copy of the soup for the main content
    main_soup = BeautifulSoup(str(soup), "html.parser")

    # Remove headers, footers, and tables from the main_soup
    for element in main_soup.find_all(["header", "footer", "table"]):
        element.extract()

    # Get all paragraphs and headings in the main content
    paragraphs = main_soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])

    # Sample a few paragraphs to use for presence tests
    if paragraphs:
        sampled_paragraphs = random.sample(paragraphs, min(5, len(paragraphs)))

        for paragraph in sampled_paragraphs:
            text = paragraph.get_text().strip()
            # Only create tests for paragraphs with sufficient content
            if text and len(text) > 20:
                tests.append(
                    {
                        "pdf": pdf_filename,
                        "page": page_num,
                        "id": f"{pdf_id}_text_{uuid.uuid4().hex[:8]}",
                        "type": TestType.PRESENT.value,
                        "text": text[:200],  # Limit to 200 chars to keep tests manageable
                        "max_diffs": 10,
                    }
                )

    # Generate some TextOrderTests for content that should appear in a specific order
    if len(paragraphs) >= 2:
        # Extract all text from the main content
        all_text = " ".join([p.get_text().strip() for p in paragraphs])

        # Use syntok to segment the text into sentences
        sentences = []
        for paragraph in process(all_text):
            for sentence in paragraph:
                # Convert token sequence to string and clean it
                sentence_text = " ".join([token.value for token in sentence]).strip()
                if sentence_text and len(sentence_text) > 10 and len(sentence_text) < 100:
                    sentences.append(sentence_text)

        # Create TextOrderTests from pairs of sentences that are at least 3 sentences apart
        # to ensure they're from different parts of the document
        if len(sentences) >= 5:
            num_tests = min(3, len(sentences) // 5)
            for _ in range(num_tests):
                # Get two random indices with sufficient distance between them
                i = random.randint(0, len(sentences) - 4)
                j = random.randint(i + 3, min(i + 10, len(sentences) - 1))

                first_sentence = sentences[i]
                second_sentence = sentences[j]

                tests.append(
                    {
                        "pdf": pdf_filename,
                        "page": page_num,
                        "id": f"{pdf_id}_order_{uuid.uuid4().hex[:8]}",
                        "type": TestType.ORDER.value,
                        "before": first_sentence,
                        "after": second_sentence,
                        "max_diffs": 10,
                    }
                )

    return tests


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

        # Generate tests from the HTML content
        tests = generate_tests_from_html(html_content, pdf_id, page_num)

        # Save tests to a JSONL file
        tests_dir = os.path.join(args.output_dir, "tests")
        os.makedirs(tests_dir, exist_ok=True)
        tests_path = os.path.join(tests_dir, f"{pdf_id}_page{page_num}_tests.jsonl")
        with open(tests_path, "w") as f:
            for test in tests:
                f.write(json.dumps(test) + "\n")
        print(f"Generated {len(tests)} tests for {pdf_id}, page {page_num}")

        # Extract the page and save as PDF
        pdf_path = os.path.join(templates_dir, f"{pdf_id}_page{page_num}.pdf")
        if not extract_page_from_pdf(local_pdf_path, pdf_path, page_num):
            print(f"Failed to extract page {page_num} from {local_pdf_path}")

        # Render PDF using Playwright if not skipped
        playwright_pdf_path = None
        render_success = False

        if not args.skip_playwright:
            playwright_pdf_path = os.path.join(templates_dir, f"{pdf_id}_page{page_num}_playwright.pdf")

            try:
                # Get PNG dimensions
                png_width, png_height = get_png_dimensions_from_base64(image_base64)

                # Run the async function in the synchronous context
                render_success = asyncio.run(render_pdf_with_playwright(html_content, playwright_pdf_path, png_width, png_height))

                if render_success:
                    print(f"Successfully rendered with Playwright: {playwright_pdf_path}")
                else:
                    print(f"Failed to render as a single page PDF: {playwright_pdf_path}")
                    # Remove the tests if we couldn't render a proper single-page PDF
                    if os.path.exists(tests_path):
                        os.remove(tests_path)
                        print(f"Removed tests for {pdf_id} due to rendering failure")
                    playwright_pdf_path = None
            except Exception as e:
                print(f"Failed to render with Playwright: {e}")
                playwright_pdf_path = None
                render_success = False

        # If playwright rendering failed and was required, return None to skip this test
        if not args.skip_playwright and not render_success:
            return None

        return {
            "pdf_id": pdf_id,
            "s3_path": s3_path,
            "page_number": page_num,
            "html_path": html_path,
            "pdf_path": pdf_path,
            "playwright_pdf_path": playwright_pdf_path,
            "tests_path": tests_path,
            "num_tests": len(tests),
        }
    except Exception as e:
        print(f"Error processing {s3_path}: {e}")
        return None
    finally:
        # Clean up temp directory for this PDF
        if os.path.exists(temp_pdf_dir):
            subprocess.run(["rm", "-rf", temp_pdf_dir])


def main():
    parser = argparse.ArgumentParser(description="Convert PDFs to HTML templates and render with Playwright")
    parser.add_argument("--input_list", required=True, help="Path to a file containing S3 paths to PDFs")
    parser.add_argument("--output_dir", required=True, help="Directory to store extracted pages and tests")
    parser.add_argument("--temp_dir", default="/tmp/mine_tables", help="Directory for temporary files")
    parser.add_argument("--max_tests", type=int, default=100, help="Maximum number of tests to generate")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel threads to use")
    parser.add_argument("--api_key", help="Claude API key (or set ANTHROPIC_API_KEY environment variable)")
    parser.add_argument("--skip_playwright", action="store_true", help="Skip Playwright PDF rendering")
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

    # Print summary of Playwright rendering results
    playwright_success = sum(1 for r in results if r and r.get("playwright_pdf_path"))
    if not args.skip_playwright:
        print(f"Playwright PDF rendering: {playwright_success}/{len(results)} successful")

    # Print summary of generated tests
    total_tests = sum(r.get("num_tests", 0) for r in results if r)
    print(f"Generated a total of {total_tests} tests across {len(results)} templates")

    # Optional: Collect and display test type statistics
    if total_tests > 0:
        # Count the tests by type from a sample of result files
        test_types = {"present": 0, "absent": 0, "table": 0, "order": 0}
        for r in results[: min(10, len(results))]:
            if r and r.get("tests_path"):
                try:
                    with open(r.get("tests_path"), "r") as f:
                        for line in f:
                            test = json.loads(line)
                            test_type = test.get("type", "")
                            if test_type in test_types:
                                test_types[test_type] += 1
                except Exception as e:
                    print(f"Error reading test file {r.get('tests_path')}: {e}")

        # Print test type distribution for the sample
        print("Test type distribution (from sample):")
        for test_type, count in test_types.items():
            print(f"  - {test_type}: {count} tests")


if __name__ == "__main__":
    main()
