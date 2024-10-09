import argparse
import os
import json
import hashlib
import logging
from collections import defaultdict
from typing import Optional
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from pypdf import PdfReader
from cached_path import cached_path
from smart_open import smart_open

from dataclasses import dataclass

# Import your existing modules if necessary
# from dolma_refine.evaluate.metrics import DocumentEditSimilarity
# from dolma_refine.evaluate.segmenters import SpacySegmenter
# from dolma_refine.evaluate.aligners import HirschbergAligner

@dataclass(frozen=True)
class NormalizedEntry:
    s3_path: str
    pagenum: int
    text: Optional[str]
    finish_reason: Optional[str]
    error: Optional[str] = None

    @staticmethod
    def from_goldkey(goldkey: str, **kwargs):
        s3_path = goldkey[:goldkey.rindex("-")]
        page_num = int(goldkey[goldkey.rindex("-") + 1:])
        return NormalizedEntry(s3_path, page_num, **kwargs)

    @property
    def goldkey(self):
        return f"{self.s3_path}-{self.pagenum}"

def normalize_json_entry(data: dict) -> NormalizedEntry:
    if "outputs" in data:
        # Birr case
        if data["outputs"] is None:
            text = None
            finish_reason = None
        else:
            text = data["outputs"][0]["text"]
            finish_reason = data["outputs"][0]["finish_reason"]

        # Try to parse the structured output if possible
        try:
            if text is not None:
                parsed_content = json.loads(text)
                text = parsed_content["natural_text"]
        except json.JSONDecodeError:
            pass
        
        return NormalizedEntry.from_goldkey(
            goldkey=data["custom_id"],
            text=text,
            finish_reason=finish_reason,
            error=data.get("completion_error", None)
        )
    else:
        # OpenAI case
        try:
            # Attempt to parse the JSON content from OpenAI's response
            parsed_content = json.loads(data["response"]["body"]["choices"][0]["message"]["content"])
            return NormalizedEntry.from_goldkey(
                goldkey=data["custom_id"],
                text=parsed_content["natural_text"],
                finish_reason=data["response"]["body"]["choices"][0]["finish_reason"]
            )
        except json.JSONDecodeError:
            # Fallback if content is not valid JSON
            return NormalizedEntry.from_goldkey(
                goldkey=data["custom_id"],
                text=data["response"]["body"]["choices"][0]["message"]["content"],
                finish_reason=data["response"]["body"]["choices"][0]["finish_reason"]
            )

def parse_s3_path(s3_path):
    if not s3_path.startswith('s3://'):
        raise ValueError('Invalid S3 path')
    s3_path = s3_path[5:]
    bucket_name, _, key = s3_path.partition('/')
    return bucket_name, key

def process_document(s3_path, entries, output_dir):
    """
    Processes a single document:
    - Downloads the PDF
    - Validates and assembles text
    - Writes the output JSON if successful
    - Returns processing results for aggregation
    """
    try:
        # Download the PDF locally
        pdf_local_path = cached_path(s3_path, quiet=True)
        pdf = PdfReader(pdf_local_path)
        total_pages_in_pdf = len(pdf.pages)
    except Exception as e:
        logging.error(f"Error downloading or reading PDF {s3_path}: {e}")
        return {
            'processed': 1,
            'successful_documents': 0,
            'successful_pages': 0,
            'total_pages': 0
        }

    # Build mapping from pagenum to entry
    entry_by_pagenum = {entry.pagenum: entry for entry in entries}

    valid_entries = []
    missing_pages = []
    errors = []

    # Iterate from 1 to total_pages_in_pdf inclusive
    for page_num in range(1, total_pages_in_pdf + 1):
        entry = entry_by_pagenum.get(page_num)
        if entry is None:
            missing_pages.append(page_num)
        elif entry.error is not None or entry.finish_reason != 'stop':
            errors.append(entry)
        else:
            valid_entries.append(entry)

    if not missing_pages and not errors:
        # Assemble text
        valid_entries_sorted = sorted(valid_entries, key=lambda x: x.pagenum)
        text = '\n'.join(entry.text for entry in valid_entries_sorted if entry.text)

        # Generate a filename based on the s3_path
        doc_hash = hashlib.md5(s3_path.encode('utf-8')).hexdigest()
        output_filename = os.path.join(output_dir, f'{doc_hash}.json')

        output_data = {
            'source': s3_path,
            'total_pages': total_pages_in_pdf,
            'text': text
        }

        try:
            with open(output_filename, 'w') as f_out:
                json.dump(output_data, f_out)
            return {
                'processed': 1,
                'successful_documents': 1,
                'successful_pages': len(valid_entries),
                'total_pages': total_pages_in_pdf
            }
        except Exception as e:
            logging.error(f"Error writing output file {output_filename}: {e}")
            return {
                'processed': 1,
                'successful_documents': 0,
                'successful_pages': 0,
                'total_pages': total_pages_in_pdf
            }
    else:
        missing = [page for page in missing_pages]
        error_pages = [e.pagenum for e in errors]
        logging.info(f'Document {s3_path} has missing pages: {missing} or errors in pages: {error_pages}')
        return {
            'processed': 1,
            'successful_documents': 0,
            'successful_pages': len(valid_entries),
            'total_pages': total_pages_in_pdf
        }

def main():
    parser = argparse.ArgumentParser(description='Process finished birr inference outputs into dolma docs')
    parser.add_argument('s3_path', help='S3 path to the directory containing JSON or JSONL files')
    parser.add_argument('--output_dir', default='output', help='Directory to save the output files')
    parser.add_argument('--max_workers', type=int, default=8, help='Maximum number of worker threads')
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(filename='processing.log', level=logging.INFO, format='%(asctime)s %(message)s')

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize S3 client
    s3 = boto3.client('s3')
    bucket_name, prefix = parse_s3_path(args.s3_path)

    # List all .json and .jsonl files in the specified S3 path
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    files = []
    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('.json') or key.endswith('.jsonl'):
                    files.append(key)

    # Build documents mapping
    documents = defaultdict(list)

    print("Processing JSON files and building documents mapping...")
    for key in tqdm(files):
        file_s3_path = f's3://{bucket_name}/{key}'
        try:
            with smart_open(file_s3_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    entry = normalize_json_entry(data)
                    documents[entry.s3_path].append(entry)
        except Exception as e:
            logging.error(f"Error processing file {file_s3_path}: {e}")

    total_documents = len(documents)
    successful_documents = 0
    total_pages = 0
    successful_pages = 0

    print("Processing documents with ThreadPoolExecutor...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Prepare futures
        future_to_s3 = {
            executor.submit(
                process_document,
                s3_path,
                entries,
                args.output_dir
            ): s3_path for s3_path, entries in documents.items()
        }

        # Use tqdm to display progress
        for future in tqdm(as_completed(future_to_s3), total=len(future_to_s3)):
            try:
                result = future.result()
                successful_documents += result.get('successful_documents', 0)
                successful_pages += result.get('successful_pages', 0)
                total_pages += result.get('total_pages', 0)
            except Exception as e:
                s3_path = future_to_s3[future]
                logging.error(f"Error processing document {s3_path}: {e}")

    print(f'Total documents: {total_documents}')
    print(f'Successful documents: {successful_documents}')
    print(f'Total pages: {total_pages}')
    print(f'Successful pages: {successful_pages}')

if __name__ == '__main__':
    main()
