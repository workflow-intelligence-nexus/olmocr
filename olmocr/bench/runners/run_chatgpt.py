import os
import tempfile
import base64
import torch

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.data.anchor import get_anchor_text
from olmocr.data.prompts import build_openai_silver_data_prompt

from openai import OpenAI
 

def run_chatgpt(pdf_path: str, page_num: int=1, model: str='gpt-4o-2024-08-06') -> str:
    """
    Convert page of a PDF file to markdown using GOT-OCR.
    
    This function renders the first page of the PDF to an image, runs OCR on that image,
    and returns the OCR result as a markdown-formatted string.
    
    Args:
        pdf_path (str): The local path to the PDF file.
        
    Returns:
        str: The OCR result in markdown format.
    """
    # Convert the first page of the PDF to a base64-encoded PNG image.
    base64image = render_pdf_to_base64png(pdf_path, page_num=page_num, target_longest_image_dim=1024)
    anchor_text = get_anchor_text(local_pdf_path, page, pdf_engine="pdfreport")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        messages= [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_openai_silver_data_prompt(anchor_text)},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ],
                }
            ],
        temperature=0.1,
        max_tokens=3000,
        logprobs=True,
        top_logprobs=5,
        response_format=openai_response_format_schema()
    )
    print(response)
    
    return result


