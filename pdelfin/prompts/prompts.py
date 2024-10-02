import re

# This is the prompt we use for getting chat gpt 4o to convert documents into our silver training data
def build_openai_silver_data_prompt(base_text: str) -> str:
    return (
        f"Below is the image of one page of a PDF document, as well as some raw textual content that was previously extracted for it that includes position information for each image and block of text (The origin [0x0] of the coordinates is in the lower left corner of the image). "
        f"Just return the plain text representation of this document as if you were reading it naturally.\n"
        f"Turn equations into a LaTeX representation, and tables into markdown format. Remove the headers and footers, but keep references and footnotes.\n"
        f"Read any natural handwriting.\n"
        f"This is likely one page out of several in the document, so be sure to preserve any sentences that come from the previous page, or continue onto the next page, exactly as they are.\n"
        f"If there is no text at all that you think you should read, just output [NO TEXT].\n"
        f"If the page has no English text on it at all, just output [NO ENGLISH TEXT].\n"
        f"Do not hallucinate.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
    )


# This is a base prompt that will be used for training and running the fine tuned model
# It's simplified from the prompt which was used to generate the silver data, and can change from dataset to dataset
def build_finetuning_prompt(base_text: str) -> str:
    return (
        f"Below is the image of one page of a document, as well as some raw textual content that was previously extracted for it. "
        f"Just return the plain text representation of this document as if you were reading it naturally.\n"
        f"Do not hallucinate.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
    )


def extract_raw_text(prompt: str) -> str:
    pattern = r"RAW_TEXT_START\s*\n(.*?)\nRAW_TEXT_END"

    # Use re.DOTALL to ensure that the dot matches newline characters
    match = re.search(pattern, prompt, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        raise ValueError("Prompt does not contain raw text")
