# This script runs olmocr bench
# It will take in as arguments a folder, and scan it for .jsonl files which contain the various rules and properties that we will check
# We will then validate the json files to make sure they are all valid
# Then, each other folder in there (besides /pdfs) represents a pipeline tool that we will evaluate
# We will validate that each one of those contains a .md file coressponding to its parse for every .pdf in the /pdfs folder
# Then, we will read each one, and check if they pass against all the rules

import argparse
import os
import json
import glob

def validate_jsonl_file(jsonl_path: str, all_pdf_files: list[str]):
    """
    Validate a .jsonl file line by line to ensure each line is valid JSON
    and has the expected fields for the rules.
    """
    all_pdf_basenames = [os.path.basename(p) for p in all_pdf_files]

    rules = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                # You could decide if blank lines are okay or not
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} in {jsonl_path}: {e}")

            # Basic checks to ensure required keys exist (pdf, id, type, etc.)
            if "pdf" not in data or "id" not in data or "type" not in data:
                raise ValueError(f"Missing required fields in line {line_num} of {jsonl_path}: {data}")
            
            rule_id = data["id"]
            
            # Make sure the document referenced exists
            if data["pdf"] not in all_pdf_basenames:
                raise ValueError(f"Missing pdf {data['pdf']} referenced by {rule_id} in {jsonl_path} line {line_num}")

            # Additional validations depending on type
            rule_type = data["type"]
            if rule_type in ("present", "absent"):
                if "text" not in data:
                    raise ValueError(f"'text' field required for rule type '{rule_type}' in {jsonl_path} line {line_num}")
            elif rule_type == "order":
                # Check that anchor is present, and that either 'before' or 'after' is present
                if "anchor" not in data:
                    raise ValueError(f"'anchor' field required for rule type 'order' in {jsonl_path} line {line_num}")
                if not ("before" in data or "after" in data):
                    raise ValueError(f"'before' or 'after' required for rule type 'order' in {jsonl_path} line {line_num}")

            # If everything looks good, add to the rules list
            rules.append(data)

    return rules

def run_rule(rule, md_file_path: str) -> bool:
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Example skeleton checks
    rule_type = rule["type"]

    if rule_type == "present":
        return rule["text"] in md_content
    elif rule_type == "absent":
        return rule["text"] not in md_content
    elif rule_type == "order":
        # Check ordering constraints, e.g. anchor vs. before/after
        # This is highly dependent on how you want to define "order" in text.
        # For instance, you might do:
        anchor = rule.get("anchor", "")
        before = rule.get("before", "")
        after = rule.get("after", "")
        # ...
        # Implement your logic to confirm that anchor occurs before or after certain text
        return False

    raise NotImplementedError

def main():
    parser = argparse.ArgumentParser(description="Run OLMOCR Bench.")
    parser.add_argument("--input_folder",
                        default=os.path.join(os.path.dirname(__file__), "sample_data"),
                        help="Path to the folder containing .jsonl files, /pdfs folder, and pipeline tool subfolders.")
    args = parser.parse_args()

    input_folder = args.input_folder
    pdf_folder = os.path.join(input_folder, "pdfs")

    # Find all pdf files in the data folder 
    assert os.path.exists(pdf_folder), "/pdfs folder must exist in your data directory"
    all_pdf_files = list(glob.glob(os.path.join(pdf_folder, "*.pdf")))
    assert all_pdf_files, f"No PDF files found in  {pdf_folder}"

    # Find .jsonl files and validate them
    jsonl_files = glob.glob(os.path.join(input_folder, "*.jsonl"))
    assert jsonl_files, f"No .jsonl files found in {input_folder}."

    all_rules = []
    for jsonl_path in jsonl_files:
        print(f"Validating JSONL file: {jsonl_path}")
        try:
            rules = validate_jsonl_file(jsonl_path, all_pdf_files)
            all_rules.extend(rules)
        except ValueError as e:
            print(f"Validation error in {jsonl_path}: {e}")
            return

    # Now, find all of the other folders in the input folder, those become the candidates
    # Each candidate will then run each rule on its content

    # At the end, print a summary of the score (number of passing rules) for each candidate
    # Make it a pretty interface, similar to what pytest does

if __name__ == "__main__":
    main()
