import os
import csv
import json
import argparse
import re
import collections
import random
import sqlite3

from urllib.parse import urlparse
from tqdm import tqdm

def parse_pdf_hash(pretty_pdf_path: str) -> str:
    """
    Given a string like "s3://ai2-s2-pdfs/4342/6a12ffc2ffa73f5258eb66095659beae9522.pdf-32",
    extract the hash ("43426a12ffc2ffa73f5258eb66095659beae9522").
    Returns None if not found.
    """
    pattern = r"s3://ai2-s2-pdfs/([a-f0-9]{4})/([a-f0-9]+)\.pdf-\d+"
    match = re.match(pattern, pretty_pdf_path)
    if match:
        return match.group(1) + match.group(2)
    return None

def cache_athena_csv_to_db(athena_csv_path: str) -> str:
    """
    Cache the Athena CSV file into an SQLite database.
    Returns the path to the SQLite database.
    """
    db_path = athena_csv_path + ".db"

    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA synchronous = OFF;")
        cursor.execute("PRAGMA journal_mode = MEMORY;")
        

        # Create the table
        cursor.execute("""
            CREATE TABLE pdf_mapping (
                pdf_hash TEXT PRIMARY KEY,
                uri TEXT
            )
        """)

        # Insert data from CSV in batches of 1000 rows
        with open(athena_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            batch = []
            for row in tqdm(reader):
                batch.append((row["distinct_pdf_hash"], row["uri"]))
                if len(batch) == 1000:
                    cursor.executemany("INSERT INTO pdf_mapping (pdf_hash, uri) VALUES (?, ?)", batch)
                    conn.commit()
                    batch = []

            # Insert remaining rows
            if batch:
                cursor.executemany("INSERT INTO pdf_mapping (pdf_hash, uri) VALUES (?, ?)", batch)
                conn.commit()

        conn.close()

    return db_path

def get_uri_from_db(db_path: str, pdf_hash: str) -> str:
    """
    Query the SQLite database to retrieve the URI for a given PDF hash.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT uri FROM pdf_mapping WHERE pdf_hash = ?", (pdf_hash,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def main():
    parser = argparse.ArgumentParser(
        description="Review silver dataset and provide summary statistics based on source URL and also provide a few data samples for review."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="openai_batch_data",
        help="Input folder, which is the output of the buildsilver.py script",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="openai_batch_data_summary",
        help="Output destination (folder)",
    )
    parser.add_argument(
        "--athena-csv",
        type=str,
        default="/home/ubuntu/s2pdf_url_data/c974870d-3b06-4793-9a62-d46d38e2c8b2.csv",
        help="CSV file that maps pdf_hash to uri",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="How many sample rows to include in the sample CSV",
    )

    args = parser.parse_args()

    # Cache the Athena CSV into SQLite database
    db_path = cache_athena_csv_to_db(args.athena_csv)

    # Process input JSONL files
    all_rows = []

    for filename in tqdm(os.listdir(args.input)):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(args.input, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        print("Error parsing line")
                        continue

                    custom_id = data.get("custom_id")
                    if not custom_id:
                        print("No custom_id found")
                        continue

                    pdf_hash = parse_pdf_hash(custom_id)
                    assert pdf_hash, f"Need to have a pdf_hash {custom_id}"

                    uri = get_uri_from_db(db_path, pdf_hash)

                    domain = None
                    if uri:
                        parsed = urlparse(uri)
                        domain = parsed.netloc

                    all_rows.append((custom_id, uri, domain))

    # Write output CSVs
    os.makedirs(args.output, exist_ok=True)

    output_csv_path = os.path.join(args.output, "custom_id_to_url.csv")
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["custom_id", "uri", "domain"])
        for (cid, uri, domain) in all_rows:
            writer.writerow([cid, uri if uri else "", domain if domain else ""])

    domain_counter = collections.Counter()
    for (_, _, domain) in all_rows:
        if domain:
            domain_counter[domain] += 1

    most_common_domains = domain_counter.most_common(1000)
    domain_csv_path = os.path.join(args.output, "top_1000_domains.csv")
    with open(domain_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "count"])
        for domain, count in most_common_domains:
            writer.writerow([domain, count])

    sample_size = min(args.sample_size, len(all_rows))
    sample_rows = random.sample(all_rows, sample_size) if all_rows else []
    sample_csv_path = os.path.join(args.output, "data_samples.csv")
    with open(sample_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["custom_id", "uri", "domain"])
        for (cid, uri, domain) in sample_rows:
            writer.writerow([cid, uri if uri else "", domain if domain else ""])

    print(f"Summary files written to: {args.output}")
    print(f" - Full mapping: {output_csv_path}")
    print(f" - Top domains: {domain_csv_path}")
    print(f" - Samples: {sample_csv_path}")

if __name__ == "__main__":
    main()
