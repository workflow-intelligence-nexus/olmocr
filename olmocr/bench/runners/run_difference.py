import os
import json

from openai import OpenAI

from run_gemini import run_gemini
from run_chatgpt import run_chatgpt
from olmocr.prompts.prompts import build_find_difference_prompt, PageResponse
from olmocr.data.renderpdf import render_pdf_to_base64png


def combined_output(pdf_path: str) -> str:
    chatgpt_output = run_chatgpt(pdf_path)
    gemini_output = run_gemini(pdf_path)
    return (
        f"ChatGPT OUTPUT: \n"
        f"{chatgpt_output}\n\n"
        f"Gemini OUTPUT: \n"
        f"{gemini_output}"
    )

def run_difference(pdf_path: str, page_num: int = 1, model: str = "gpt-4o-2024-08-06", temperature: float=0.1) -> str:
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
    image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num, target_longest_image_dim=2048)
    anchor_text = combined_output(pdf_path)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_find_difference_prompt(anchor_text)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ],
        temperature=temperature,
        max_tokens=3000,
        # response_format=openai_response_format_schema(),
    )

    raw_response = response.choices[0].message.content

    # assert len(response.choices) > 0
    # assert response.choices[0].message.refusal is None
    # assert response.choices[0].finish_reason == "stop"

    # data = json.loads(raw_response)
    # data = PageResponse(**data)

    return raw_response

if __name__ == "__main__":
    import argparse
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Extract text from a PDF using OpenAI OCR")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--page", type=int, default=1, help="Page number to process (default: 1)")
    parser.add_argument("--model", default="gpt-4o-2024-08-06", help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for generation")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Run the OCR function
    result = run_difference(
        pdf_path=args.pdf_path,
        page_num=args.page,
        model=args.model,
        temperature=args.temperature
    )
    
    # Print the result
    print(result)
