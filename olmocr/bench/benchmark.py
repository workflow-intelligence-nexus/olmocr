#!/usr/bin/env python3
"""
This script runs olmocr bench.
It will take as an argument a folder, and scan it for .jsonl files which contain the various rules and properties that we will check.
It will then validate the JSON files to make sure they are all valid.
Then, each other folder in there (besides /pdfs) represents a pipeline tool that we will evaluate.
We will validate that each one of those contains a .md file corresponding to its parse for every .pdf in the /pdfs folder.
Then, we will read each one, and check if they pass against all the rules.
If a rule fails, a short explanation is printed.
"""

import argparse
import os
import json
import glob
import sys
import itertools

from rapidfuzz import fuzz
from fuzzysearch import find_near_matches

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
                if "before" not in data:
                    raise ValueError(f"'before' field required for rule type 'order' in {jsonl_path} line {line_num}")
                if len(data["before"]) < 10:
                    raise ValueError(f"'before' field too short {jsonl_path} line {line_num}")
                if "after" not in data:
                    raise ValueError(f"'after' required for rule type 'order' in {jsonl_path} line {line_num}")
                if len(data["after"]) < 10:
                    raise ValueError(f"'after' field too short {jsonl_path} line {line_num}")
            else:
                raise ValueError(f"Unknown rule type '{rule_type}' in {jsonl_path} line {line_num}")

            # If everything looks good, add to the rules list
            rules.append(data)

    return rules


def run_rule(rule, md_file_path: str) -> (bool, str):
    """
    Run the given rule on the content of the provided .md file.
    Returns a tuple (passed, explanation) where 'passed' is True if the rule passes,
    and 'explanation' is a short message explaining the failure when the rule does not pass.
    """
    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
    except Exception as e:
        return (False, f"Error reading {md_file_path}: {e}")

    rule_type = rule["type"]

    if rule_type in ("present", "absent"):
        reference_query = rule["text"]
        threshold = rule.get("threshold", 1.0)

        best_ratio = fuzz.partial_ratio(reference_query, md_content) / 100.0

        if rule_type == "present":
            if best_ratio >= threshold:
                return (True, "")
            else:
                return (False, f"Expected text to be present with threshold {threshold} but best match ratio was {best_ratio:.3f}")
        else:  # absent
            if best_ratio < threshold:
                return (True, "")
            else:
                return (False, f"Expected text to be absent with threshold {threshold} but best match ratio was {best_ratio:.3f}")
    elif rule_type == "order":
        # Implement a simple ordering check: ensure that the anchor text appears,
        # and if 'before' is specified, it must appear before the anchor;
        # if 'after' is specified, it must appear after the anchor.
        before = rule.get("before")
        after = rule.get("after")
        threshold = rule.get("threshold", 1.0)

        max_l_dist = round((1.0 - threshold) * len(before))

        before_matches = find_near_matches(before, md_content, max_l_dist=max_l_dist)
        after_matches = find_near_matches(after, md_content, max_l_dist=max_l_dist)

        if not before_matches:
            return (False, f"'before' search text '{before[:40]}...' does not appear in parse with max_l_dist {max_l_dist}")
        
        if not after_matches:
            return (False, f"'after' search text '{after[:40]}...' does not appear in parse with max_l_dist {max_l_dist}")
        
        # Go through each combination of matches and see if there exists one where the before .start is sooner than the after .start
        for before_match, after_match in itertools.product(before_matches, after_matches):
            if before_match.start < after_match.start:
                return (True, "")

        return (False, f"Could not find a place in the text where '{before[:40]}...' appears before '{after[:40]}...'.")

    else:
        raise NotImplementedError(f"Rule type '{rule_type}' is not implemented.")


def evaluate_candidate(candidate_folder: str, all_rules: list, pdf_basenames: list[str]):
    """
    For the candidate folder (pipeline tool output), first validate that it contains
    a .md file for every PDF in the pdf folder. Then, run each rule against the corresponding
    .md file.

    Returns a tuple:
        (num_passed, total_rules, candidate_errors, rule_failures, rule_type_breakdown)
    where:
      - candidate_errors is a list of error strings (e.g. missing files or exceptions)
      - rule_failures is a list of rule failure messages (a rule returning False is not an error)
      - rule_type_breakdown is a dict with rule type as key and a tuple (passed, total) as value

    NOTE: A rule returning False is not considered an 'error' but simply a rule failure.
          Only exceptions and missing files are treated as candidate errors.
          The rule_type_breakdown is added for a detailed breakdown of performance per rule type.
    """
    candidate_errors = []
    rule_failures = []
    rule_type_breakdown = {}  # key: rule type, value: [passed_count, total_count]
    candidate_name = os.path.basename(candidate_folder)
    num_passed = 0
    total_rules = 0

    # Validate that a .md file exists for every PDF.
    for pdf_name in pdf_basenames:
        # Change .pdf extension to .md (assumes pdf_name ends with .pdf)
        md_name = os.path.splitext(pdf_name)[0] + ".md"
        md_path = os.path.join(candidate_folder, md_name)
        if not os.path.exists(md_path):
            candidate_errors.append(f"Candidate '{candidate_name}' is missing {md_name} corresponding to {pdf_name}.")

    # If there are missing .md files, we don't run the rules.
    if candidate_errors:
        return (0, len(all_rules), candidate_errors, rule_failures, rule_type_breakdown)

    # Evaluate rules. Each rule references a PDF (e.g., "doc1.pdf"), and we expect the candidate to have "doc1.md".
    for rule in all_rules:
        rule_type = rule["type"]
        # Initialize breakdown counts for this rule type if not already
        if rule_type not in rule_type_breakdown:
            rule_type_breakdown[rule_type] = [0, 0]
        rule_type_breakdown[rule_type][1] += 1  # increment total count

        pdf_name = rule["pdf"]
        md_name = os.path.splitext(pdf_name)[0] + ".md"
        md_path = os.path.join(candidate_folder, md_name)
        total_rules += 1
        try:
            passed, explanation = run_rule(rule, md_path)
            if passed:
                num_passed += 1
                rule_type_breakdown[rule_type][0] += 1  # increment passed count
            else:
                # A rule returning False is recorded as a rule failure, not an error.
                rule_failures.append(f"Rule {rule.get('id')} on {md_name} failed: {explanation}")
        except Exception as e:
            # Exceptions are considered candidate errors.
            candidate_errors.append(f"Error running rule {rule.get('id')} on {md_name}: {e}")

    return (num_passed, total_rules, candidate_errors, rule_failures, rule_type_breakdown)

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
        num_passed, total_rules, candidate_errors, rule_failures, rule_type_breakdown = evaluate_candidate(candidate, all_rules, pdf_basenames)
        summary.append((candidate_name, num_passed, total_rules, candidate_errors, rule_failures, rule_type_breakdown))
        print(f"\nCandidate: {candidate_name}")
        if candidate_errors:
            for err in candidate_errors:
                print(f"  [ERROR] {err}")
        else:
            if rule_failures:
                for fail in rule_failures:
                    print(f"  [FAIL] {fail}")
            print(f"  Passed {num_passed} out of {total_rules} rules.")

    # Print a final summary (if only rule failures occurred, we output the score and breakdown)
    print("\n" + "="*50)
    print("Final Summary:")
    for candidate_name, num_passed, total_rules, candidate_errors, _, rule_type_breakdown in summary:
        if candidate_errors:
            status = "FAILED (errors)"
        else:
            status = f"{num_passed / total_rules * 100:0.1f}%"
        print(f"{candidate_name:20s} : {num_passed:3d}/{total_rules:3d} rules passed - {status}")
        print("  Breakdown by rule type:")
        for rtype, counts in rule_type_breakdown.items():
            passed_count, total_count = counts
            percentage = passed_count / total_count * 100 if total_count else 0
            print(f"    {rtype:8s}: {passed_count:2d}/{total_count:2d} rules passed ({percentage:0.1f}%)")

    print("="*50)

if __name__ == "__main__":
    main()
