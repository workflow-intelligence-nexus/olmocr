import os
import argparse
import tempfile
import base64
import torch

from olmocr.data.renderpdf import render_pdf_to_base64png

from transformers import AutoModel, AutoTokenizer

# Load GOT-OCR model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    'ucaslcl/GOT-OCR2_0', trust_remote_code=True
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(
    'ucaslcl/GOT-OCR2_0',
    trust_remote_code=True,
    use_safetensors=True,
    revision="979938bf89ccdc949c0131ddd3841e24578a4742",
    pad_token_id=tokenizer.eos_token_id
)
model = model.eval().to(device)


def run(pdf_folder):
    """
    Convert all PDF files in the specified folder to markdown using GOT-OCR.
    Each page of a PDF is converted to an image and processed with OCR.
    The markdown files are saved in a folder called "got_ocr" located alongside the pdf_folder.
    
    :param pdf_folder: Path to the folder containing PDF files.
    """
    # Resolve absolute paths and prepare destination folder
    pdf_folder = os.path.abspath(pdf_folder)
    parent_dir = os.path.dirname(pdf_folder)
    destination_folder = os.path.join(parent_dir, "got_ocr")
    os.makedirs(destination_folder, exist_ok=True)

    # List all PDF files in the folder
    pdf_files = [
        os.path.join(pdf_folder, filename)
        for filename in os.listdir(pdf_folder)
        if filename.lower().endswith(".pdf")
    ]

    for pdf_path in pdf_files:
        print(f"Processing {pdf_path} ...")

        base64image = render_pdf_to_base64png(pdf_path, page_num=1, target_longest_image_dim=1024)
       
        # Save the image temporarily as a JPEG file
        with tempfile.NamedTemporaryFile("wb", suffix=".png", delete=False) as tmp:
            tmp.write(base64.b64decode(base64image))
        
        # Run GOT-OCR on the saved image
        # The OCR result is assumed to be a plain text string.
        res = model.chat(tokenizer, tmp.name, ocr_type='ocr')

        # Clean up the temporary image file
        os.remove(tmp.name)

        # Create the markdown filename by replacing .pdf with .md
        file_name = os.path.basename(pdf_path).replace('.pdf', '.md')
        output_path = os.path.join(destination_folder, file_name)
        
        with open(output_path, "w", encoding="utf-8") as fout:
            fout.write(res)
        
        print(f"Saved markdown to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert all PDF files in a folder to markdown using GOT-OCR and save them to a sibling 'marker' folder."
    )
    parser.add_argument(
        "pdf_folder",
        type=str,
        help="Path to the folder containing PDF files (e.g., '/path/to/pdfs')"
    )
    args = parser.parse_args()
    run(args.pdf_folder)
