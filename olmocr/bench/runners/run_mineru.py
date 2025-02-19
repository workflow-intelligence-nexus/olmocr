import os
import shutil
import argparse

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod


def run(pdf_folder):
    """
    Convert all PDF files in the specified folder to markdown using MinerU.
    For each PDF file, the script outputs markdown files along with visual and JSON outputs.
    The outputs are saved in a folder called "mineru" (with an "images" subfolder) 
    located in the same parent directory as pdf_folder.
    
    :param pdf_folder: Path to the folder containing PDF files.
    """
    # Resolve absolute paths
    pdf_folder = os.path.abspath(pdf_folder)
    parent_dir = os.path.dirname(pdf_folder)
    output_folder = os.path.join(parent_dir, "mineru")
    image_output_folder = os.path.join(output_folder, "images")

    # Create output directories if they don't exist
    os.makedirs(image_output_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Initialize writers (same for all PDFs)
    image_writer = FileBasedDataWriter(image_output_folder)
    md_writer = FileBasedDataWriter(output_folder)

    # List all PDF files in the provided folder
    pdf_files = [
        os.path.join(pdf_folder, filename)
        for filename in os.listdir(pdf_folder)
        if filename.lower().endswith(".pdf")
    ]

    for pdf_path in pdf_files:
        print(f"Processing {pdf_path}...")
        # Get file name without suffix for naming outputs
        pdf_file_name = os.path.basename(pdf_path)
        name_without_suff = pdf_file_name.split(".")[0]

        # Read the PDF file bytes
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(pdf_path)

        # Create dataset instance
        ds = PymuDocDataset(pdf_bytes)

        # Inference: decide whether to run OCR mode based on dataset classification
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)

        # Generate markdown content; the image directory is the basename of the images output folder
        image_dir_basename = os.path.basename(image_output_folder)
        md_content = pipe_result.get_markdown(image_dir_basename)

        # Dump markdown file
        md_file_name = f"{name_without_suff}.md"
        pipe_result.dump_md(md_writer, md_file_name, image_dir_basename)

        # Remove useless image folder
        shutil.rmtree(image_output_folder)

        print(f"Finished processing {pdf_file_name}. Outputs saved to {output_folder}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert all PDF files in a folder to markdown and related outputs using MinerU."
    )
    parser.add_argument(
        "pdf_folder",
        type=str,
        help="Path to the folder containing PDF files (e.g., '/path/to/pdfs')"
    )
    args = parser.parse_args()
    run(args.pdf_folder)
