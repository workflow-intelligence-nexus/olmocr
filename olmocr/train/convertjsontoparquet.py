# Script to generate parquet dataset files to upload to hugging face
# Input is a dataset location /data/jakep/pdfdata/openai_batch_data_v5_1_iabooks_train_done/*.json
# Each json line has a custom id that looks like {"custom_id": "s3://ai2-s2-pdfs/de80/a57e6c57b45796d2e020173227f7eae44232.pdf-1", ... more data}

# Fix this script so that it works, and that it will take a path to an input dataset, and sqllite database location
# And then it will build a parquet file with rows that look like: "id", "url", "page_number", "response"
# Where Id will be the output of parse_pdf_hash plus "-" plus the page number
# The url will be the result of get_uri_from_db
# Rresponse will be NormalizedEntry.text

import argparse
import glob
import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Optional

from tqdm import tqdm
import pandas as pd


def parse_pdf_hash(pretty_pdf_path: str) -> Optional[str]:
    """
    Extracts a hash from a pretty PDF S3 URL.
    For example, given:
      s3://ai2-s2-pdfs/de80/a57e6c57b45796d2e020173227f7eae44232.pdf-1
    it will return "de80a57e6c57b45796d2e020173227f7eae44232".
    """
    pattern = r"s3://ai2-s2-pdfs/([a-f0-9]{4})/([a-f0-9]+)\.pdf"
    match = re.match(pattern, pretty_pdf_path)
    if match:
        return match.group(1) + match.group(2)
    return None


def get_uri_from_db(db_path: str, pdf_hash: str) -> Optional[str]:
    """
    Looks up the URL for the given pdf_hash in the sqlite database.
    Assumes there is a table called 'pdf_mapping' with a column 'uri'.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT uri FROM pdf_mapping WHERE pdf_hash = ?", (pdf_hash,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


@dataclass(frozen=True)
class NormalizedEntry:
    s3_path: str
    pagenum: int
    text: Optional[str]
    finish_reason: Optional[str]
    error: Optional[str] = None

    @staticmethod
    def from_goldkey(goldkey: str, **kwargs):
        """
        Constructs a NormalizedEntry from a goldkey string.
        The goldkey is expected to be of the format: 
          <s3_path>-<page_number>
        """
        s3_path = goldkey[: goldkey.rindex("-")]
        page_num = int(goldkey[goldkey.rindex("-") + 1 :])
        return NormalizedEntry(s3_path, page_num, **kwargs)

    @property
    def goldkey(self):
        return f"{self.s3_path}-{self.pagenum}"


def normalize_json_entry(data: dict) -> NormalizedEntry:
    """
    Normalizes a JSON entry from any of the supported formats.
    It supports:
      - Birr: looks for an "outputs" field.
      - Already normalized entries: if they contain s3_path, pagenum, etc.
      - OpenAI: where the response is in data["response"]["body"]["choices"].
      - SGLang: where the response is in data["response"]["choices"].
    """
    if "outputs" in data:
        # Birr case
        if data["outputs"] is None:
            text = None
            finish_reason = None
        else:
            text = data["outputs"][0]["text"]
            finish_reason = data["outputs"][0]["finish_reason"]

        return NormalizedEntry.from_goldkey(
            goldkey=data["custom_id"],
            text=text,
            finish_reason=finish_reason,
            error=data.get("completion_error", None),
        )
    elif all(field in data for field in ["s3_path", "pagenum", "text", "error", "finish_reason"]):
        # Already normalized
        return NormalizedEntry(**data)
    elif "response" in data and "body" in data["response"] and "choices" in data["response"]["body"]:
            return NormalizedEntry.from_goldkey(
                goldkey=data["custom_id"],
                text=data["response"]["body"]["choices"][0]["message"]["content"],
                finish_reason=data["response"]["body"]["choices"][0]["finish_reason"],
            )


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Parquet dataset file for HuggingFace upload."
    )
    parser.add_argument(
        "input_dataset",
        help="Input dataset file pattern (e.g., '/data/jakep/pdfdata/openai_batch_data_v5_1_iabooks_train_done/*.json')",
    )
    parser.add_argument(
        "db_path", help="Path to the SQLite database file."
    )
    parser.add_argument(
        "--output", default="output.parquet", help="Output Parquet file path."
    )
    args = parser.parse_args()

    rows = []
    files = glob.glob(args.input_dataset)
    print(f"Found {len(files)} files matching pattern: {args.input_dataset}")

    for file_path in tqdm(files):
        print(f"Processing file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON at {file_path}:{line_num} - {e}")
                    continue

                try:
                    normalized = normalize_json_entry(data)
                except Exception as e:
                    print(f"Error normalizing entry at {file_path}:{line_num} - {e}")
                    continue

                # Use the s3_path from the normalized entry to extract the pdf hash.
                pdf_hash = parse_pdf_hash(normalized.s3_path)
                if pdf_hash is None:
                    print(
                        f"Could not parse pdf hash from {normalized.s3_path} at {file_path}:{line_num}"
                    )
                    continue

                # The output id is the pdf hash plus '-' plus the page number.
                combined_id = f"{pdf_hash}-{normalized.pagenum}"

                # Look up the corresponding URL from the sqlite database.
                url = get_uri_from_db(args.db_path, pdf_hash)
                if url is None:
                    print(
                        f"No URL found in DB for pdf hash {pdf_hash} at {file_path}:{line_num}"
                    )
                    continue

                row = {
                    "id": combined_id,
                    "url": url,
                    "page_number": normalized.pagenum,
                    "response": normalized.text,
                }
                rows.append(row)

        break

    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(args.output, index=False)
        print(f"Successfully wrote {len(df)} rows to {args.output}")
    else:
        print("No rows to write. Exiting.")


if __name__ == "__main__":
    main()
