import json
import os

from mistralai import Mistral


def run_mistral(pdf_path: str, page_num: int = 1) -> str:
    """
    Convert page of a PDF file to markdown using the mistral OCR api
    https://docs.mistral.ai/capabilities/document/

    Args:
        pdf_path (str): The local path to the PDF file.

    Returns:
        str: The OCR result in markdown format.
    """
    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)

    with open(pdf_path, "rb") as pf:
        uploaded_pdf = client.files.upload(
            file={
                "file_name": os.path.basename(pdf_path),
                "content": pf,
            },
            purpose="ocr"
        )  

    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

    ocr_response = client.ocr.process(
        model="mistral-ocr-2503",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        }
    )

    client.files.delete(file_id=uploaded_pdf.id)

    return ocr_response.pages[0].markdown
