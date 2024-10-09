import subprocess
import base64
import io
from pypdf import PdfReader

from PIL import Image


def render_pdf_to_base64png(local_pdf_path: str, page: int, target_longest_image_dim: int=2048):
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
            str(target_longest_image_dim * 72 / longest_dim), # 72 pixels per point is the conversion factor
            local_pdf_path,
        ],
        timeout=120,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert pdftoppm_result.returncode == 0, pdftoppm_result.stderr
    return base64.b64encode(pdftoppm_result.stdout).decode("utf-8")


def render_pdf_to_base64webp(local_pdf_path: str, page: int, target_longest_image_dim: int=1024):
    base64_png = render_pdf_to_base64png(local_pdf_path, page, target_longest_image_dim)
    
    png_image = Image.open(io.BytesIO(base64_png.encode("utf-8")))
    webp_output = io.BytesIO()
    png_image.save(webp_output, format="WEBP")
        
    return base64.b64encode(webp_output.getvalue()).decode("utf-8")