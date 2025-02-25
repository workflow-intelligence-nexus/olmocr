import os
import json
import base64
from anthropic import Anthropic
from olmocr.prompts.anchor import get_anchor_text
from olmocr.prompts.prompts import build_silver_data_prompt, response_format_schema, PageResponse
from olmocr.data.renderpdf import render_pdf_to_base64png

def run_claude(pdf_path: str, page_num: int = 1, model: str = "claude-3-7-sonnet-20250219", temperature: float=0.1) -> str:
    """
    Convert page of a PDF file to markdown using Claude OCR.
    This function renders the specified page of the PDF to an image, runs OCR on that image,
    and returns the OCR result as a markdown-formatted string.
    
    Args:
        pdf_path (str): The local path to the PDF file.
        page_num (int): The page number to process (starting from 1).
        model (str): The Claude model to use.
        temperature (float): The temperature parameter for generation.
        
    Returns:
        str: The OCR result in markdown format.
    """
    # Convert the specified page of the PDF to a base64-encoded PNG image
    image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num, target_longest_image_dim=2048)
    
    # Get anchor text for the page
    anchor_text = get_anchor_text(pdf_path, page_num, pdf_engine="pdfreport")
    
    # Initialize the Claude client
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Create the message with the prompt and image
    response = client.messages.create(
        model=model,
        max_tokens=3000,
        temperature=temperature,
        system=build_silver_data_prompt(anchor_text),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64
                        }
                    },
                    {"type": "json_object", "schema": response_format_schema()}
                ]
            }
        ],
        
    )
    
    # Extract and validate the response
    raw_response = response.content#[0].text
    print(raw_response)
    # Parse the JSON response
    data = json.loads(raw_response)
    data = PageResponse(**data)
    
    return data.natural_text

if __name__ == "__main__":
    import argparse
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Extract text from a PDF using Claude OCR")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--page", type=int, default=1, help="Page number to process (default: 1)")
    parser.add_argument("--model", default="claude-3-7-sonnet-20250219", help="Claude model to use")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for generation")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Run the OCR function
    result = run_claude(
        pdf_path=args.pdf_path,
        page_num=args.page,
        model=args.model,
        temperature=args.temperature
    )
    
    # Print the result
    print(result)