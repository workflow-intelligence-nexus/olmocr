# import json
# import random
# import re
# import argparse

# def read_file_and_fix_json(file_path):
#     """
#     Read the entire file and use a regex to extract JSON objects.
    
#     The expected format in the file is (possibly spanning multiple lines):
#     {"image": "1", "text": "some text here ..."}
    
#     This function returns a dict mapping pdf names (like "1.pdf") to the full text.
#     """
#     print(f"Reading file: {file_path}")
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             content = file.read()
        
#         document_texts = {}
#         # This regex looks for: {"image": "<number>", "text": "<text>"}
#         # Using DOTALL so that the text portion can include newline characters.
#         document_pattern = r'\{"image":\s*"(\d+)"\s*,\s*"text":\s*"(.*?)"\}'
#         matches = re.findall(document_pattern, content, re.DOTALL)
        
#         for image_num, text in matches:
#             pdf_name = f"{image_num}.pdf"
#             # Replace newlines (and subsequent spaces) with a single space
#             cleaned_text = re.sub(r'\n\s*', ' ', text)
#             document_texts[pdf_name] = cleaned_text
#             print(f"Found text for {pdf_name}")
        
#         return document_texts
#     except Exception as e:
#         print(f"Error reading file: {e}")
#         return {}

# def get_random_sentence(text):
#     """
#     Extract a random complete sentence from the provided text.
    
#     This function splits the text into sentences using punctuation (".", "!", or "?")
#     followed by whitespace.
#     """
#     text = text.strip()
#     # Split text into sentences. The regex uses a positive lookbehind to keep the punctuation.
#     sentences = re.split(r'(?<=[.!?])\s+', text)
#     # Remove any empty strings
#     sentences = [s for s in sentences if s]
#     if not sentences:
#         return ""
#     return random.choice(sentences)

# def process_jsonl(input_path, output_path):
#     """
#     Process the malformed JSONL file:
    
#     1. Extract complete JSON objects using a regex.
#     2. For each document, extract one random complete sentence.
#     3. Create a processed JSON object containing the fields:
#        - "pdf": the original PDF filename (e.g. "1.pdf")
#        - "page": set to 1
#        - "id": constructed as "<pdf>_processed01"
#        - "type": "present"
#        - "max_diffs": 2
#        - "text": the randomly chosen sentence
#        - "case_sensitive": true
#        - "first_n": null
#        - "last_n": null
#     4. Write the processed objects as a JSONL file.
#     """
#     documents = read_file_and_fix_json(input_path)
#     if not documents:
#         print("No documents found in the file.")
#         return
    
#     output_lines = []
#     for pdf_name, text in documents.items():
#         random_sentence = get_random_sentence(text)
#         processed_obj = {
#             "pdf": pdf_name,
#             "page": 1,
#             "id": f"{pdf_name}_processed01",
#             "type": "present",
#             "max_diffs": 2,
#             "text": random_sentence,
#             "case_sensitive": True,
#             "first_n": None,
#             "last_n": None
#         }
#         output_lines.append(processed_obj)
    
#     with open(output_path, 'w', encoding='utf-8') as outfile:
#         for obj in output_lines:
#             outfile.write(json.dumps(obj) + "\n")
    
#     print(f"Processed {len(output_lines)} documents. Output written to {output_path}.")

# def main():
#     parser = argparse.ArgumentParser(
#         description="Process a malformed JSONL file and extract a random sentence from each document."
#     )
#     parser.add_argument("input_file", help="Path to the input JSONL file")
#     parser.add_argument("output_file", help="Path to the output JSONL file")
#     args = parser.parse_args()
    
#     process_jsonl(args.input_file, args.output_file)

# if __name__ == '__main__':
#     main()

import json
import random
import re
import argparse

def read_file_and_fix_json(file_path):
    """
    Read the entire file and use a regex to extract JSON objects.
    
    The expected format in the file is (possibly spanning multiple lines):
      {"image": "1", "text": "some text here ..."}
    
    This function returns a dict mapping pdf names (like "1.pdf") to the full text.
    """
    print(f"Reading file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        document_texts = {}
        # This regex looks for: {"image": "<number>", "text": "<text>"}
        # DOTALL allows the text to span across multiple lines.
        document_pattern = r'\{"image":\s*"(\d+)"\s*,\s*"text":\s*"(.*?)"\}'
        matches = re.findall(document_pattern, content, re.DOTALL)
        
        for image_num, text in matches:
            pdf_name = f"{image_num}.pdf"
            # Replace newlines (and following spaces) with a single space
            cleaned_text = re.sub(r'\n\s*', ' ', text)
            document_texts[pdf_name] = cleaned_text
            print(f"Found text for {pdf_name}")
        
        return document_texts
    except Exception as e:
        print(f"Error reading file: {e}")
        return {}

def split_into_sentences(text):
    """
    Split text into sentences. A sentence is assumed to end with a period, exclamation mark, or question mark,
    followed by whitespace.
    """
    text = text.strip()
    # Using positive lookbehind to keep the punctuation in sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out empty strings
    sentences = [s for s in sentences if s]
    return sentences

def process_jsonl_for_order(input_path, output_path):
    """
    Process the input JSONL file and generate ordering test objects.
    
    For each document, randomly select two complete sentences:
      - "before": a randomly chosen sentence.
      - "after": a different randomly chosen sentence.
      
    If only one sentence is present, both fields are set to that sentence.
    
    The output JSON object follows this format:
      {
        "pdf": "<pdf_name>",
        "page": 1,
        "id": "<pdf_name>_processed02",
        "type": "order",
        "before": "<sentence>",
        "after": "<sentence>",
        "max_diffs": 3
      }
    """
    documents = read_file_and_fix_json(input_path)
    if not documents:
        print("No documents found in the file.")
        return
    
    output_objects = []
    for pdf_name, text in documents.items():
        sentences = split_into_sentences(text)
        if not sentences:
            continue

        if len(sentences) < 2:
            # If only one sentence, use it for both before and after.
            before = after = sentences[0]
        else:
            # Randomly pick two different sentences.
            before, after = random.sample(sentences, 2)
        
        ordering_obj = {
            "pdf": pdf_name,
            "page": 1,
            "id": f"{pdf_name}_processed02",  # e.g., "1.pdf_processed02"
            "type": "order",
            "before": before,
            "after": after,
            "max_diffs": 3
        }
        output_objects.append(ordering_obj)
        print(f"Created ordering test for {pdf_name}")
    
    # Write the processed JSON objects as JSON lines.
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for obj in output_objects:
            outfile.write(json.dumps(obj) + "\n")
    
    print(f"Processed {len(output_objects)} ordering tests. Output written to {output_path}.")

def main():
    parser = argparse.ArgumentParser(
        description="Generate ordering tests from a malformed JSONL file by extracting complete sentences."
    )
    parser.add_argument("input_file", help="Path to the input JSONL file")
    parser.add_argument("output_file", help="Path to the output JSONL file for ordering tests")
    args = parser.parse_args()
    
    process_jsonl_for_order(args.input_file, args.output_file)

if __name__ == '__main__':
    main()