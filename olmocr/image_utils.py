import os
import tempfile
import subprocess


def convert_image_to_pdf_bytes(image_file: str) -> bytes:
    try:
        # Run img2pdf and capture its stdout directly as bytes
        result = subprocess.run(
            ["img2pdf", image_file],
            check=True,
            capture_output=True
        )
        
        # Return the stdout content which contains the PDF data
        return result.stdout
    
    except subprocess.CalledProcessError as e:
        # Raise error with stderr information if the conversion fails
        raise RuntimeError(f"Error converting image to PDF: {e.stderr.decode('utf-8')}")


def is_png(file_path):
    try:
        with open(file_path, "rb") as f:
            header = f.read(8)
            return header == b"\x89PNG\r\n\x1a\n"
    except Exception as e:
        print(f"Error: {e}")
        return False


def is_jpeg(file_path):
    try:
        with open(file_path, 'rb') as f:
            header = f.read(2)
            return header == b'\xff\xd8'
    except Exception as e:
        print(f"Error: {e}")
        return False