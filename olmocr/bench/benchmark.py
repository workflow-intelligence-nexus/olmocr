#!/usr/bin/env python3
"""
This script runs olmocr bench.
It will take as an argument a folder, and scan it for .jsonl files which contain the various rules and properties that we will check.
It will then validate the JSON files to make sure they are all valid.
Then, each other folder in there (besides /pdfs) represents a pipeline tool that we will evaluate.
We will validate that each one of those contains at least one .md file (or repeated generations, e.g. _1.md, _2.md, etc.)
corresponding to its parse for every .pdf in the /pdfs folder.
Then, we will read each one, and check if they pass against all the rules.
If a rule fails on some of the repeats, a short explanation is printed.
The final score is averaged over the repeated generations.
"""

import argparse
import glob
import itertools
import json
import os
import sys
from typing import Tuple, List, Dict

from fuzzysearch import find_near_matches
from rapidfuzz import fuzz

from .datatypes import BasePDFTest, TextPresenceTest, TextOrderTest, TestType, load_tests

def run_rule(rule: BasePDFTest, md_content: str) -> Tuple[bool, str]:
    """
    Run the given rule on the content of the provided .md file.
    Returns a tuple (passed, explanation) where 'passed' is True if the rule passes,
    and 'explanation' is a short message explaining the failure when the rule does not pass.
    """
    rule_type = rule.type

    if rule_type in (TestType.PRESENT.value, TestType.ABSENT.value):
        # This is a TextPresenceTest
        if not isinstance(rule, TextPresenceTest):
            return (False, f"Rule type mismatch: expected TextPresenceTest but got {type(rule).__name__}")
        
        reference_query = rule.text
        threshold = rule.threshold
        best_ratio = fuzz.partial_ratio(reference_query, md_content) / 100.0
        
        if rule_type == TestType.PRESENT.value:
            if best_ratio >= threshold:
                return (True, "")
            else:
                return (False, f"Expected '{reference_query[:40]}...' with threshold {threshold} but best match ratio was {best_ratio:.3f}")
        else:  # absent
            if best_ratio < threshold:
                return (True, "")
            else:
                return (False, f"Expected absence of '{reference_query[:40]}...' with threshold {threshold} but best match ratio was {best_ratio:.3f}")
    
    elif rule_type == TestType.ORDER.value:
        # This is a TextOrderTest
        if not isinstance(rule, TextOrderTest):
            return (False, f"Rule type mismatch: expected TextOrderTest but got {type(rule).__name__}")
        
        before = rule.before
        after = rule.after
        threshold = rule.threshold
        max_l_dist = round((1.0 - threshold) * len(before))
        
        before_matches = find_near_matches(before, md_content, max_l_dist=max_l_dist)
        after_matches = find_near_matches(after, md_content, max_l_dist=max_l_dist)
        
        if not before_matches:
            return (False, f"'before' search text '{before[:40]}...' not found with max_l_dist {max_l_dist}")
        if not after_matches:
            return (False, f"'after' search text '{after[:40]}...' not found with max_l_dist {max_l_dist}")
        
        for before_match, after_match in itertools.product(before_matches, after_matches):
            if before_match.start < after_match.start:
                return (True, "")
        
        return (False, f"Could not find a location where '{before[:40]}...' appears before '{after[:40]}...'.")
    
    else:
        raise NotImplementedError(f"Rule type '{rule_type}' is not implemented.")


def evaluate_candidate(candidate_folder: str, all_rules: List[BasePDFTest], pdf_basenames: List[str]) -> Tuple[float, int, List[str], List[str], Dict[str, List[float]]]:
    """
    For the candidate folder (pipeline tool output), validate that it contains at least one .md file
    (i.e. repeated generations like _1.md, _2.md, etc.) for every PDF in the pdf folder.
    Then, run each rule against all corresponding .md files and average the results.

    Returns a tuple:
      (overall_score, total_rules, candidate_errors, rule_failures, rule_type_breakdown)

      - overall_score: Average fraction of rules passed (averaged over repeats and rules).
      - total_rules: Total number of rules evaluated.
      - candidate_errors: List of candidate errors (e.g. missing files).
      - rule_failures: List of failure messages for rules not passing on all repeats.
      - rule_type_breakdown: Dictionary mapping rule type to list of average pass ratios for rules of that type.
    """
    candidate_errors = []
    rule_failures = []
    rule_type_breakdown = {}  # key: rule type, value: list of average pass ratios
    candidate_name = os.path.basename(candidate_folder)

    # Map each PDF to its corresponding MD repeats (e.g., doc1_1.md, doc1_2.md, etc.)
    pdf_to_md_files = {}
    for pdf_name in pdf_basenames:
        md_base = os.path.splitext(pdf_name)[0]
        md_pattern = os.path.join(candidate_folder, f"{md_base}_*.md")
        md_files = glob.glob(md_pattern)
        if not md_files:
            candidate_errors.append(f"Candidate '{candidate_name}' is missing MD repeats for {pdf_name} (expected files matching {md_base}_*.md).")
        else:
            pdf_to_md_files[pdf_name] = md_files

    if candidate_errors:
        return (0.0, len(all_rules), candidate_errors, rule_failures, rule_type_breakdown)

    total_rule_score = 0.0

    # Evaluate each rule. Each rule references a PDF (e.g., "doc1.pdf") so we get all its MD repeats.
    for rule in all_rules:
        rule_type = rule.type
        if rule_type not in rule_type_breakdown:
            rule_type_breakdown[rule_type] = []
        pdf_name = rule.pdf
        md_base = os.path.splitext(pdf_name)[0]
        md_files = pdf_to_md_files.get(pdf_name, [])
        if not md_files:
            continue  # Should not occur due to earlier check.
        repeat_passes = 0
        num_repeats = 0
        explanations = []
        for md_path in md_files:
            num_repeats += 1
            try:
                with open(md_path, "r", encoding="utf-8") as f:
                    md_content = f.read()
            except Exception as e:
                candidate_errors.append(f"Error reading {md_path}: {e}")
                continue

            try:
                passed, explanation = run_rule(rule, md_content)
                if passed:
                    repeat_passes += 1
                else:
                    explanations.append(explanation)
            except Exception as e:
                candidate_errors.append(f"Error running rule {rule.id} on {md_path}: {e}")
                explanations.append(str(e))
        
        rule_avg = repeat_passes / num_repeats if num_repeats > 0 else 0.0
        total_rule_score += rule_avg
        if rule_avg < 1.0:
            rule_failures.append(
                f"Rule {rule.id} on {md_base} average pass ratio: {rule_avg:.3f} ({repeat_passes}/{num_repeats} repeats passed). "
                f"Example explanation: {explanations[0] if explanations else 'No explanation'}"
            )
        rule_type_breakdown[rule_type].append(rule_avg)

    overall_score = total_rule_score / len(all_rules) if all_rules else 0.0
    return (overall_score, len(all_rules), candidate_errors, rule_failures, rule_type_breakdown)


def main():
    parser = argparse.ArgumentParser(description="Run OLMOCR Bench.")
    parser.add_argument(
        "--input_folder",
        default=os.path.join(os.path.dirname(__file__), "sample_data"),
        help="Path to the folder containing .jsonl files, /pdfs folder, and pipeline tool subfolders.",
    )
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

    # Find and validate .jsonl files in the input folder
    jsonl_files = glob.glob(os.path.join(input_folder, "*.jsonl"))
    if not jsonl_files:
        print(f"Error: No .jsonl files found in {input_folder}.", file=sys.stderr)
        sys.exit(1)

    # Load and concatenate all test rules from JSONL files
    all_tests = []
    for jsonl_path in jsonl_files:
        tests = load_tests(jsonl_path)
        all_tests.extend(tests)

    if not all_tests:
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
        overall_score, total_rules, candidate_errors, rule_failures, rule_type_breakdown = evaluate_candidate(candidate, all_tests, pdf_basenames)
        summary.append((candidate_name, overall_score, total_rules, candidate_errors, rule_failures, rule_type_breakdown))
        print(f"\nCandidate: {candidate_name}")
        if candidate_errors:
            for err in candidate_errors:
                print(f"  [ERROR] {err}")
        else:
            if rule_failures:
                for fail in rule_failures:
                    print(f"  [FAIL] {fail}")
            print(f"  Average Score: {overall_score * 100:.1f}% over {total_rules} rules.")

    # Print final summary with breakdown by rule type
    print("\n" + "=" * 50)
    print("Final Summary:")
    for candidate_name, overall_score, total_rules, candidate_errors, _, rule_type_breakdown in summary:
        if candidate_errors:
            status = "FAILED (errors)"
        else:
            status = f"{overall_score * 100:0.1f}%"
        print(f"{candidate_name:20s} : Average Score: {overall_score * 100:0.1f}% over {total_rules:3d} rules - {status}")
        print("  Breakdown by rule type:")
        for rtype, scores in rule_type_breakdown.items():
            if scores:
                avg = sum(scores) / len(scores) * 100
            else:
                avg = 0.0
            print(f"    {rtype:8s}: {avg:0.1f}% average pass rate over {len(scores)} rules")
    print("=" * 50)


if __name__ == "__main__":
    main()