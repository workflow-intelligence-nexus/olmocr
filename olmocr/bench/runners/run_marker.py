import os
import time
import argparse


from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


def run(pdf_folder):
    """
    Convert all PDF files in the specified folder to markdown.
    The markdown files are saved in a folder called "marker" located in the same root directory as the pdf_folder.

    :param pdf_folder: Path to the folder containing PDF files.
    """
    # Resolve absolute paths
    pdf_folder = os.path.abspath(pdf_folder)
    parent_dir = os.path.dirname(pdf_folder)
    destination_folder = os.path.join(parent_dir, "marker")

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # List all PDF files in the provided folder
    pdf_files = [
        os.path.join(pdf_folder, filename)
        for filename in os.listdir(pdf_folder)
        if filename.lower().endswith(".pdf")
    ]

    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )


    for pdf_path in pdf_files:
        rendered = converter(pdf_path)
        # Create the markdown filename by replacing the .pdf extension with .md
        text, _, images = text_from_rendered(rendered)

        file_name = os.path.basename(pdf_path).replace('.pdf', '.md')
        output_path = os.path.join(destination_folder, file_name)

        with open(output_path, "w") as fout:
            fout.write(text)

        print(f"Saved markdown to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert all PDF files in a folder to markdown and save them to a sibling 'marker' folder."
    )
    parser.add_argument(
        "pdf_folder",
        type=str,
        help="Path to the folder containing PDF files (e.g., '/path/to/pdfs')"
    )
    args = parser.parse_args()
    run(args.pdf_folder)

