import argparse
import json
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import logging

import smart_open

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


def is_s3_path(path):
    """Check if the given path is an S3 path."""
    return str(path).startswith('s3://')


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


def process_file(input_file: str, output_file: str, rewrite_prompt_str: bool):
    """
    Process a single JSONL file: read, transform, and write to output.

    Args:
        input_file (str): Path or URL to the input JSONL file.
        output_file (str): Path or URL to the output JSONL file.
    """
    processed_count = 0
    error_count = 0

    try:
        with smart_open.open(input_file, 'r', encoding='utf-8') as infile, \
                smart_open.open(output_file, 'w', encoding='utf-8') as outfile:

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
                    outfile.write(json.dumps(transformed) + '\n')
                    processed_count += 1
                else:
                    error_count += 1

        logging.info(f"Processed '{input_file}': {processed_count} records transformed, {error_count} errors.")
    except Exception as e:
        logging.error(f"Failed to process file {input_file}: {e}")


def construct_output_file_path(input_file_path, input_dir, output_dir):
    """
    Given an input file path, input directory, and output directory,
    construct the corresponding output file path.

    Args:
        input_file_path (str): Path to the input file.
        input_dir (str): Path to the input directory.
        output_dir (str): Path to the output directory.

    Returns:
        str: Path to the output file.
    """
    input_file = Path(input_file_path)
    input_dir_path = Path(input_dir)
    relative_path = input_file.relative_to(input_dir_path)

    if is_s3_path(output_dir):
        # For S3 output paths, construct the S3 URL manually
        output_file_path = output_dir.rstrip('/') + '/' + str(relative_path).replace('\\', '/')
    else:
        # For local output paths
        output_file_path = str(Path(output_dir) / relative_path)
    return output_file_path


def list_input_files(input_dir):
    """
    List all JSONL files in the input directory.

    Args:
        input_dir (str): Path to the input directory.

    Returns:
        list: List of input file paths.
    """
    if is_s3_path(input_dir):
        # Use smart_open's s3 functionality to list files
        import boto3
        s3 = boto3.resource('s3')
        bucket_name = input_dir.split('s3://')[1].split('/')[0]
        prefix = '/'.join(input_dir.split('s3://')[1].split('/')[1:])
        bucket = s3.Bucket(bucket_name)
        files = []
        for obj in bucket.objects.filter(Prefix=prefix):
            if obj.key.endswith('.jsonl'):
                files.append(f's3://{bucket_name}/{obj.key}')
        return files
    else:
        input_dir_path = Path(input_dir)
        return [str(p) for p in input_dir_path.glob('*.jsonl')]


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Transform JSONL files by extracting and renaming specific fields."
    )
    parser.add_argument(
        '--rewrite_finetuning_prompt',
        action='store_true',
        default=False,
        help="Rewrites the input prompt from standard OPENAI instruction format into our finetuned format"
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Path to the input directory containing JSONL files. Can be a local path or S3 URL.'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Path to the output directory where transformed JSONL files will be saved. Can be a local path or S3 URL.'
    )
    parser.add_argument(
        '--jobs', '-j',
        type=int,
        default=20,
        help='Number of parallel jobs to run (default: 20).'
    )
    args = parser.parse_args()

    input_dir = args.input_dir.rstrip('/')
    output_dir = args.output_dir.rstrip('/')
    max_jobs = args.jobs

    # List input files
    input_files = list_input_files(input_dir)

    if not input_files:
        logging.warning(f"No JSONL files found in '{input_dir}'. Exiting.")
        sys.exit(0)

    logging.info(f"Found {len(input_files)} JSONL files to process.")

    # Prepare tasks for parallel processing
    tasks = []
    for input_file in input_files:
        output_file = construct_output_file_path(input_file, input_dir, output_dir)
        tasks.append((input_file, output_file))

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_jobs) as executor:
        future_to_file = {
            executor.submit(process_file, input_file, output_file, args.rewrite_finetuning_prompt): input_file
            for input_file, output_file in tasks
        }

        for future in as_completed(future_to_file):
            input_file = future_to_file[future]
            try:
                future.result()
            except Exception as exc:
                logging.error(f"File {input_file} generated an exception: {exc}")

    logging.info("All files have been processed.")


if __name__ == "__main__":
    main()
