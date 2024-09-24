from jinja2 import Template
import random
import os
import subprocess
import tempfile
import boto3
import base64
import io

from urllib.parse import urlparse
from PIL import Image
from tqdm import tqdm

session = boto3.Session(profile_name='s2')
s3_client = session.client('s3')


def render_pdf_to_base64png(s3_path, page):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        pdf_path = tmp_pdf.name
        bucket, key = s3_path.replace("s3://", "").split('/', 1)
        s3_client.download_file(bucket, key, pdf_path)

        # Render the PDF to an image, and display it in the first position
        pdftoppm_result = subprocess.run(
                    ["pdftoppm",
                        "-png",
                        "-f", str(page),
                        "-l", str(page),
                        pdf_path],
                        timeout=120,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert pdftoppm_result.returncode == 0, pdftoppm_result.stderr

        png_image = Image.open(io.BytesIO(pdftoppm_result.stdout))
        webp_output = io.BytesIO()
        png_image.save(webp_output, format="WEBP")
        
        image_base64 = base64.b64encode(webp_output.getvalue()).decode("utf-8")

        return image_base64


def create_review_html(data, filename="review_page.html"):
    # Load the Jinja2 template from the file
    with open(os.path.join(os.path.dirname(__file__), "evalhtml_template.html"), "r") as f:
        template = Template(f.read())
    
    entries = []
    for i, entry in tqdm(enumerate(data)):
        # Randomly decide whether to display gold on the left or right
        if random.choice([True, False]):
            left_text, right_text = entry["gold_text"], entry["eval_text"]
            left_alignment, right_alignment = entry["alignment"], entry["alignment"]
            left_class, right_class = "gold", "eval"
        else:
            left_text, right_text = entry["eval_text"], entry["gold_text"]
            left_alignment, right_alignment = entry["alignment"], entry["alignment"]
            left_class, right_class = "eval", "gold"

        # Convert newlines to <p> tags for proper formatting
        left_text = "<p>" + left_text.replace("\n", "</p><p>") + "</p>"
        right_text = "<p>" + right_text.replace("\n", "</p><p>") + "</p>"

        parsed_url = urlparse(entry["s3_path"])
        bucket = parsed_url.netloc
        s3_key = parsed_url.path.lstrip('/')
        signed_pdf_link = s3_client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": s3_key}, ExpiresIn=604800)

        # Create a dictionary for each entry
        entries.append({
            "entry_id": i,
            "page_image": render_pdf_to_base64png(entry["s3_path"], entry["page"]),
            "s3_path": entry["s3_path"],
            "page": entry["page"],
            "signed_pdf_link": signed_pdf_link,
            "left_text": left_text,
            "right_text": right_text,
            "left_alignment": left_alignment,
            "right_alignment": right_alignment,
            "left_class": left_class,
            "right_class": right_class,
            "gold_class": "gold" if left_class == "gold" else "eval",
            "eval_class": "eval" if right_class == "eval" else "gold"
        })

    # Render the template with the entries
    final_html = template.render(entries=entries)

    # Write the HTML content to the specified file
    with open(filename, "w") as f:
        f.write(final_html)

    print(f"HTML file '{filename}' created successfully!")
