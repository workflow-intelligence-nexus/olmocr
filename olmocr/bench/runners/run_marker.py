import os
import time
import argparse


from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


def run_marker(pdf_path: str, page_num: int=1) -> str:
    # List all PDF files in the provided folder
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )

    for pdf_path in pdf_files:
        rendered = converter(pdf_path)
        # Create the markdown filename by replacing the .pdf extension with .md
        text, _, images = text_from_rendered(rendered)

        return text

