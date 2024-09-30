# Converts data that was built by "buildsilver.py" into something you can feed to the mise/birr batch inference pipeline
# to efficiently generate eval samples with against a local model

import argparse
import json
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import logging

from pdelfin.prompts import build_finetuning_prompt


def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def transform_json_object(obj):
    """
    Transform a single JSON object by extracting and renaming specific fields.

    Args:
        obj (dict): Original JSON object.

    Returns:
        dict: Transformed JSON object.
    """
    try:
        transformed = {
            "custom_id": obj["custom_id"],
            "chat_messages": obj["body"]["messages"],
            "temperature": obj["body"]["temperature"],
            "max_tokens": obj["body"]["max_tokens"]
        }
        return transformed
    except KeyError as e:
        logging.error(f"Missing key {e} in object: {obj.get('custom_id', 'unknown')}")
        return None


def process_file(input_file: Path, output_file: Path, rewrite_prompt_str: bool):
    """
    Process a single JSONL file: read, transform, and write to output.

    Args:
        input_file (Path): Path to the input JSONL file.
        output_file (Path): Path to the output JSONL file.
    """
    processed_count = 0
    error_count = 0

    try:
        with input_file.open('r', encoding='utf-8') as infile, \
             output_file.open('w', encoding='utf-8') as outfile:
            
            for line_number, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error in file {input_file} at line {line_number}: {e}")
                    error_count += 1
                    continue

                transformed = transform_json_object(obj)

                if transformed is not None and rewrite_prompt_str:
                    pattern = r"RAW_TEXT_START\s*\n(.*?)\nRAW_TEXT_END"

                    # Use re.DOTALL to ensure that the dot matches newline characters
                    match = re.search(pattern, transformed["chat_messages"][0]["content"][0]["text"], re.DOTALL)

                    if match:
                        raw_page_text = match.group(1).strip()
                        transformed["chat_messages"][0]["content"][0]["text"] = build_finetuning_prompt(raw_page_text)

                if transformed is not None:
                    json.dump(transformed, outfile)
                    outfile.write('\n')
                    processed_count += 1
                else:
                    error_count += 1

        logging.info(f"Processed '{input_file.name}': {processed_count} records transformed, {error_count} errors.")
    except Exception as e:
        logging.error(f"Failed to process file {input_file}: {e}")

def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Transform JSONL files by extracting and renaming specific fields."
    )
    parser.add_argument(
        '--rewrite_finetuning_prompt',
        action='store_true',
        default=False,
        help="Rewrites the input prompt from standard OPENAI instruction format, into our finetuned format"
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Path to the input directory containing JSONL files.'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Path to the output directory where transformed JSONL files will be saved.'
    )
    parser.add_argument(
        '--jobs', '-j',
        type=int,
        default=20,
        help='Number of parallel jobs to run (default: 20).'
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    max_jobs = args.jobs

    # Validate input directory
    if not input_dir.exists() or not input_dir.is_dir():
        logging.error(f"Input directory '{input_dir}' does not exist or is not a directory.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather all JSONL files in the input directory
    jsonl_files = list(input_dir.glob('*.jsonl'))

    if not jsonl_files:
        logging.warning(f"No JSONL files found in '{input_dir}'. Exiting.")
        sys.exit(0)

    logging.info(f"Found {len(jsonl_files)} JSONL files to process.")

    # Prepare tasks for parallel processing
    tasks = []
    with ProcessPoolExecutor(max_workers=max_jobs) as executor:
        future_to_file = {
            executor.submit(process_file, file, output_dir / file.name, args.rewrite_finetuning_prompt): file
            for file in jsonl_files
        }

        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                future.result()
            except Exception as exc:
                logging.error(f"File {file.name} generated an exception: {exc}")

    logging.info("All files have been processed.")

if __name__ == "__main__":
    main()
