#!/usr/bin/env python3
"""
mine_math.py - Extract and validate math equations from candidate files and TeX bases.

This script processes candidate files and corresponding TeX files:
1. Candidate files are located in /math_data/[candidate] folder and have names like:
   [tex file basename]_pg[pagenum]_repeat1.md
2. TeX base files are found in /math_data/pdfs.
3. For each candidate file, the candidate text is searched for within the corresponding TeX file
   using fuzzy matching (find_near_matches) to get the matching substring directly.
4. Math equations are then extracted from the matched content and validated with KaTeX.
5. Valid equations are saved as MathTest objects in a JSONL file.

Usage:
  python mine_math.py --math_data /path/to/math_data --candidate candidate_folder --output_file math_tests.jsonl
"""

import argparse
import glob
import os
import re
import random
from typing import List, Optional, Tuple

from fuzzysearch import find_near_matches
from rapidfuzz import fuzz
from tqdm import tqdm

from olmocr.bench.tests import MathTest, save_tests
from olmocr.bench.katex.render import render_equation


def normalize_text(text: str) -> str:
    """Normalize text for better matching."""
    text = re.sub(r'\s+', " ", text)
    replacements = {
        "'": "'",
        "‚": "'",
        '"': '"',
        "„": '"',
        "＿": "_",
        "–": "-", "—": "-", "‑": "-", "‒": "-"
    }
    for fancy_char, ascii_char in replacements.items():
        text = text.replace(fancy_char, ascii_char)
    return text


def extract_tex_content(tex_file: str) -> str:
    """Extract the content from a TeX file."""
    try:
        with open(tex_file, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(tex_file, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {tex_file}: {e}")
            return ""


def extract_candidate_content(candidate_file: str) -> str:
    """Extract the content from a candidate .md file."""
    try:
        with open(candidate_file, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {candidate_file}: {e}")
        return ""


def extract_math_from_tex(tex_content: str) -> List[Tuple[str, str]]:
    """
    Extract math equations from TeX content.
    Returns list of tuples (equation_type, equation_content)
    """
    math_equations = []

    # Patterns for display math
    display_patterns = [
        (r'\$\$(.*?)\$\$', '$$'),
        (r'\\begin\{equation\}(.*?)\\end\{equation\}', 'equation'),
        (r'\\begin\{equation\*\}(.*?)\\end\{equation\*\}', 'equation*'),
        (r'\\begin\{align\}(.*?)\\end\{align\}', 'align'),
        (r'\\begin\{align\*\}(.*?)\\end\{align\*\}', 'align*'),
        (r'\\begin\{displaymath\}(.*?)\\end\{displaymath\}', 'displaymath'),
        (r'\\\[(.*?)\\\]', 'displaymath')
    ]
    # Patterns for inline math
    inline_patterns = [
        (r'\$(.*?)\$', 'inline'),
        (r'\\\((.*?)\\\)', 'inline')
    ]

    for pattern_list in [display_patterns, inline_patterns]:
        for pattern, eq_type in pattern_list:
            matches = re.finditer(pattern, tex_content, re.DOTALL)
            for match in matches:
                equation = match.group(1).strip()
                if equation and not equation.isspace():
                    math_equations.append((eq_type, equation))
    return math_equations


import numpy as np
import numba

@numba.njit
def compute_dp(candidate_arr, text_arr):
    m = candidate_arr.shape[0]
    n = text_arr.shape[0]
    dp = np.empty((m + 1, n + 1), dtype=np.int32)
    # For empty candidate, cost is 0 (can match anywhere in text)
    for j in range(n + 1):
        dp[0, j] = 0
    # When text is empty, need to delete all candidate characters.
    for i in range(1, m + 1):
        dp[i, 0] = i

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if candidate_arr[i - 1] == text_arr[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j - 1] + cost,  # substitution or match
                           dp[i - 1, j] + 1,         # deletion (from candidate)
                           dp[i, j - 1] + 1)         # insertion (in candidate)
    return dp

@numba.njit
def find_best_end(dp, m, n):
    best_distance = 1 << 30  # a large number
    best_end = 0
    for j in range(n + 1):
        if dp[m, j] < best_distance:
            best_distance = dp[m, j]
            best_end = j
    return best_end, best_distance

@numba.njit
def backtrack(dp, candidate_arr, text_arr, m, best_end):
    i = m
    j = best_end
    while i > 0:
        # Check for a diagonal move (match or substitution)
        if j > 0 and dp[i, j] == dp[i - 1, j - 1] + (0 if candidate_arr[i - 1] == text_arr[j - 1] else 1):
            i -= 1
            j -= 1
        elif dp[i, j] == dp[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return j  # start index in text

def find_matching_content(candidate_text: str, tex_content: str) -> Optional[str]:
    """
    Find the substring of tex_content that most closely matches candidate_text using
    dynamic programming accelerated by numba. Returns the matching substring if its
    normalized similarity (1 - (edit_distance / len(candidate_text))) is above 0.8,
    otherwise returns None.
    """
    candidate_norm = normalize_text(candidate_text)
    tex_norm = normalize_text(tex_content)
    
    m = len(candidate_norm)
    n = len(tex_norm)
    if m == 0 or n == 0:
        return None

    # Convert strings to numpy arrays of integer character codes.
    candidate_arr = np.empty(m, dtype=np.int32)
    for i, c in enumerate(candidate_norm):
        candidate_arr[i] = ord(c)
    text_arr = np.empty(n, dtype=np.int32)
    for j, c in enumerate(tex_norm):
        text_arr[j] = ord(c)
    
    dp = compute_dp(candidate_arr, text_arr)
    best_end, min_distance = find_best_end(dp, m, n)
    similarity = (m - min_distance) / m

    print("sim", similarity)
    if similarity < 0.7:
        return None
    start_index = backtrack(dp, candidate_arr, text_arr, m, best_end)
    return tex_norm[start_index:best_end]

def parse_candidate_filename(filename: str) -> Optional[Tuple[str, int]]:
    """
    Parse candidate filename in the format: [tex file basename]_pg[pagenum]_repeat1.md
    Returns tuple (tex_basename, page_num) or None if the format doesn't match.
    """
    basename = os.path.basename(filename)
    match = re.match(r"(.+)_pg(\d+)_repeat\d+\.md$", basename)
    if match:
        tex_basename = match.group(1)
        page_num = int(match.group(2))
        return tex_basename, page_num
    return None


def validate_equation(equation: str) -> bool:
    """
    Validate that an equation renders correctly with KaTeX.
    Returns True if the equation is valid, False otherwise.
    """
    rendered = render_equation(equation)
    return rendered is not None


def process_candidate_file(candidate_file: str, pdfs_folder: str) -> List[MathTest]:
    """
    Process a single candidate file.
    Returns a list of MathTest objects extracted from the corresponding TeX file.
    """
    print("Processing", candidate_file)
    tests = []
    parse_result = parse_candidate_filename(candidate_file)
    if not parse_result:
        print(f"Filename {candidate_file} does not match expected format.")
        return tests

    tex_basename, page_num = parse_result
    tex_file_path = os.path.join(pdfs_folder, f"{tex_basename}.tex")
    
    if not os.path.exists(tex_file_path):
        print(f"TeX file {tex_file_path} not found for candidate {candidate_file}.")
        return tests
    
    candidate_text = extract_candidate_content(candidate_file)
    tex_content = extract_tex_content(tex_file_path)
    if not tex_content:
        print(f"No content extracted from {tex_file_path}")
        return tests

    matching_tex = find_matching_content(candidate_text, tex_content)
    if not matching_tex:
        print(f"No matching TeX content found in {tex_file_path} for candidate {candidate_file}")
        return tests

    print("matching tex")
    print(matching_tex)
    print("---------")

    math_equations = extract_math_from_tex(matching_tex)
    if not math_equations:
        print(f"No math equations found in matching content for candidate {candidate_file}")
        return tests

    math_equations = [(eq_type, eq.strip()) for (eq_type, eq) in math_equations if len(eq.strip()) > 20]
    math_equations = list(set(math_equations))
    random.shuffle(math_equations)

    for i, (eq_type, equation) in enumerate(math_equations):
        if validate_equation(equation):
            test_id = f"{tex_basename}_pg{page_num}_math_{i:03d}"
            math_test = MathTest(
                id=test_id,
                pdf=f"{tex_basename}.pdf",
                page=page_num,
                type="math",
                math=equation,
            )
            tests.append(math_test)
            if len(tests) >= 10:
                break

    return tests


def main():
    parser = argparse.ArgumentParser(
        description="Extract math equations from candidate files and corresponding TeX bases."
    )
    parser.add_argument("--math_data", required=True, help="Path to math_data folder")
    parser.add_argument("--candidate", required=True, help="Candidate folder name inside math_data")
    parser.add_argument("--output_file", default="math_tests.jsonl", help="Output file for math tests in JSONL format")
    
    args = parser.parse_args()
    
    candidate_folder = os.path.join(args.math_data, args.candidate)
    pdfs_folder = os.path.join(args.math_data, "pdfs")
    
    candidate_files = glob.glob(os.path.join(candidate_folder, "*.md"))
    
    all_math_tests = []
    for candidate_file in candidate_files:
        tests = process_candidate_file(candidate_file, pdfs_folder)
        all_math_tests.extend(tests)
    
    print(f"Found {len(all_math_tests)} valid math equations from {len(candidate_files)} candidate files.")
    save_tests(all_math_tests, args.output_file)
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
