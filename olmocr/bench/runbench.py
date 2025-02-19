#!/usr/bin/env python3
"""
This script runs olmocr bench.
It will take as an argument a folder, and scan it for .jsonl files which contain the various rules and properties that we will check.
It will then validate the JSON files to make sure they are all valid.
Then, each other folder in there (besides /pdfs) represents a pipeline tool that we will evaluate.
We will validate that each one of those contains a .md file corresponding to its parse for every .pdf in the /pdfs folder.
Then, we will read each one, and check if they pass against all the rules.
"""

import argparse
import os
import json
import glob
import sys

from rapidfuzz import fuzz

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
                # Skip blank lines
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
            else:
                raise ValueError(f"Unknown rule type '{rule_type}' in {jsonl_path} line {line_num}")

            # If everything looks good, add to the rules list
            rules.append(data)

    return rules


def run_rule(rule, md_file_path: str) -> bool:
    """
    Run the given rule on the content of the provided .md file.
    Returns True if the rule passes, False otherwise.
    """
    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
    except Exception as e:
        print(f"Error reading {md_file_path}: {e}")
        return False

    rule_type = rule["type"]

    if rule_type == "present" or rule_type == "absent":
        reference_query = rule["text"]
        threshold = rule.get("threshold", 1.0)

        best_ratio = fuzz.partial_ratio(reference_query, md_content) / 100.0

        if rule_type == "present":
            return best_ratio >= threshold
        else:
            return best_ratio < threshold
    elif rule_type == "order":
        # Implement a simple ordering check: ensure that the anchor text appears,
        # and if 'before' is specified, it must appear before the anchor;
        # if 'after' is specified, it must appear after the anchor.
        anchor = rule.get("anchor", "")
        before = rule.get("before")
        after = rule.get("after")

        anchor_index = md_content.find(anchor)
        if anchor_index == -1:
            return False

        if before is not None:
            before_index = md_content.find(before)
            # If 'before' text not found or appears after (or at) the anchor, fail.
            if before_index == -1 or before_index >= anchor_index:
                return False

        if after is not None:
            after_index = md_content.find(after)
            # If 'after' text not found or appears before (or at) the anchor, fail.
            if after_index == -1 or after_index <= anchor_index:
                return False

        return True

    else:
        raise NotImplementedError(f"Rule type '{rule_type}' is not implemented.")

def evaluate_candidate(candidate_folder: str, all_rules: list, pdf_basenames: list[str]):
    """
    For the candidate folder (pipeline tool output), first validate that it contains
    a .md file for every PDF in the pdf folder. Then, run each rule against the corresponding
    .md file.
    
    Returns a tuple (num_passed, total_rules, errors) where errors is a list of strings.
    """
    errors = []
    candidate_name = os.path.basename(candidate_folder)
    num_passed = 0
    total_rules = 0

    # Validate that a .md file exists for every PDF.
    for pdf_name in pdf_basenames:
        # Change .pdf extension to .md (assumes pdf_name ends with .pdf)
        md_name = os.path.splitext(pdf_name)[0] + ".md"
        md_path = os.path.join(candidate_folder, md_name)
        if not os.path.exists(md_path):
            errors.append(f"Candidate '{candidate_name}' is missing {md_name} corresponding to {pdf_name}.")
    
    if errors:
        # If candidate fails the md file existence check, do not evaluate further.
        return (0, len(all_rules), errors)

    # Evaluate rules. Each rule references a PDF (e.g., "doc1.pdf"), and we expect the candidate to have "doc1.md".
    for rule in all_rules:
        pdf_name = rule["pdf"]
        md_name = os.path.splitext(pdf_name)[0] + ".md"
        md_path = os.path.join(candidate_folder, md_name)
        total_rules += 1
        try:
            if run_rule(rule, md_path):
                num_passed += 1
        except Exception as e:
            errors.append(f"Error running rule {rule.get('id')} on {md_name}: {e}")

    return (num_passed, total_rules, errors)

def main():
    parser = argparse.ArgumentParser(description="Run OLMOCR Bench.")
    parser.add_argument("--input_folder",
                        default=os.path.join(os.path.dirname(__file__), "sample_data"),
                        help="Path to the folder containing .jsonl files, /pdfs folder, and pipeline tool subfolders.")
    args = parser.parse_args()

    input_folder = args.input_folder
    pdf_folder = os.path.join(input_folder, "pdfs")

    # Check that the pdfs folder exists
    if not os.path.exists(pdf_folder):
        print("Error: /pdfs folder must exist in your data directory.", file=sys.stderr)
        sys.exit(1)

    # Find all pdf files in the pdf folder
    all_pdf_files = list(glob.glob(os.path.join(pdf_folder, "*.pdf")))
    if not all_pdf_files:
        print(f"Error: No PDF files found in {pdf_folder}", file=sys.stderr)
        sys.exit(1)

    # Get PDF basenames (e.g. "doc1.pdf")
    pdf_basenames = [os.path.basename(p) for p in all_pdf_files]

    # Find .jsonl files in the input folder and validate them
    jsonl_files = glob.glob(os.path.join(input_folder, "*.jsonl"))
    if not jsonl_files:
        print(f"Error: No .jsonl files found in {input_folder}.", file=sys.stderr)
        sys.exit(1)

    all_rules = []
    for jsonl_path in jsonl_files:
        print(f"Validating JSONL file: {jsonl_path}")
        try:
            rules = validate_jsonl_file(jsonl_path, all_pdf_files)
            all_rules.extend(rules)
        except ValueError as e:
            print(f"Validation error in {jsonl_path}: {e}", file=sys.stderr)
            sys.exit(1)

    if not all_rules:
        print("No valid rules found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Identify candidate pipeline folders (subdirectories of input_folder excluding /pdfs)
    candidate_folders = []
    for entry in os.listdir(input_folder):
        full_path = os.path.join(input_folder, entry)
        if os.path.isdir(full_path) and entry != "pdfs":
            candidate_folders.append(full_path)

    if not candidate_folders:
        print("Error: No candidate pipeline folders found (subdirectories besides 'pdfs').", file=sys.stderr)
        sys.exit(1)

    # Evaluate each candidate
    summary = []
    print("\nRunning rules for each candidate:")
    for candidate in candidate_folders:
        candidate_name = os.path.basename(candidate)
        num_passed, total_rules, errors = evaluate_candidate(candidate, all_rules, pdf_basenames)
        summary.append((candidate_name, num_passed, total_rules, errors))
        print(f"\nCandidate: {candidate_name}")
        if errors:
            for err in errors:
                print(f"  [ERROR] {err}")
        else:
            print(f"  Passed {num_passed} out of {total_rules} rules.")

    # Print a final summary (similar to a pytest summary)
    print("\n" + "="*50)
    print("Final Summary:")
    for candidate_name, num_passed, total_rules, errors in summary:
        if errors:
            status = "FAILED (errors)"
        else:
            status = f"{num_passed / total_rules * 100:0.1f}%"
        print(f"{candidate_name:20s} : {num_passed:3d}/{total_rules:3d} rules passed - {status}")
    print("="*50)

if __name__ == "__main__":
    main()
