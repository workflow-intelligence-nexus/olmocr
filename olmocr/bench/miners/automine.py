import os
import argparse
from difflib import SequenceMatcher
from collections import Counter

import syntok.segmenter as segmenter
import syntok.tokenizer as tokenizer

def parse_sentences(text: str) -> list[str]:
    """
    Splits a text into a list of sentence strings using syntok.
    """
    sentences = []
    for paragraph in segmenter.process(text):
        for sentence in paragraph:
            # Collect token values, stripping out empty strings
            token_values = [token.value for token in sentence if token.value.strip()]
            # Join them with a space
            sentence_str = " ".join(token_values)
            sentences.append(sentence_str)
    return sentences

def compare_votes_for_file(base_text: str, candidate_texts: list[str]) -> None:
    """
    For each sentence in the base text, finds the best matching sentence from
    each candidate text (using a similarity threshold). If any candidate sentences
    differ from the base sentence, prints the base sentence along with each unique
    variant and the number of times it was chosen.
    """
    base_sentences = parse_sentences(base_text)
    # Parse all candidate texts into lists of sentences
    candidate_sentences_list = [parse_sentences(ct) for ct in candidate_texts]

    for b_sentence in base_sentences:
        votes = []
        for c_sentences in candidate_sentences_list:
            best_ratio = 0.0
            best_candidate = None

            # Find the candidate sentence with the highest similarity to b_sentence
            for c_sentence in c_sentences:
                ratio = SequenceMatcher(None, b_sentence, c_sentence).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_candidate = c_sentence

            # Append the candidate if it passes the similarity threshold (e.g., 0.7)
            if best_ratio > 0.7 and best_candidate is not None:
                votes.append(best_candidate)

        # Only consider variants that differ from the base sentence
        variant_votes = [vote for vote in votes if vote != b_sentence]
        if variant_votes:
            print("Base Sentence:")
            print(b_sentence)
            print("Variants:")
            counts = Counter(variant_votes)
            for variant, count in counts.items():
                print(f"{count}x: {variant}")
            print("-" * 40)

def main():
    parser = argparse.ArgumentParser(
        description="Compares sentences from base and candidate texts, printing differences."
    )
    parser.add_argument(
        "--base",
        default=os.path.join(os.path.dirname(__file__), "chatgpt"),
        help="Path to the folder containing base .md files."
    )
    parser.add_argument(
        "--compare",
        default=os.path.join(os.path.dirname(__file__), "olmocr"),
        help="Path to the folder containing candidate .md files."
    )
    args = parser.parse_args()

    base_path = args.base
    compare_path = args.compare

    # Collect all .md files from the base and compare folders
    base_files = [f for f in os.listdir(base_path) if f.endswith(".md")]
    compare_files = [f for f in os.listdir(compare_path) if f.endswith(".md")]

    # Read all candidate texts at once
    candidate_texts = []
    for cf in compare_files:
        with open(os.path.join(compare_path, cf), "r", encoding="utf-8") as f:
            candidate_texts.append(f.read())

    # Process each base file and print out the vote differences
    for bf in base_files:
        base_file_path = os.path.join(base_path, bf)
        with open(base_file_path, "r", encoding="utf-8") as f:
            base_text = f.read()

        print(f"Results for base file: {bf}")
        compare_votes_for_file(base_text, candidate_texts)
        print("=" * 80)

if __name__ == "__main__":
    main()
