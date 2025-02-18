#!/usr/bin/env python3
import argparse
import boto3
import json
import random
import requests
import time

# Given a key in S3, use reservoir sampling to pick a random line from the JSONL file.
def get_random_line_from_s3(bucket, key):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=key)
    random_line = None
    count = 0
    # Iterate over each line in the file (decoded as UTF-8)
    for line in response['Body'].iter_lines():
        if not line:
            continue
        line_str = line.decode('utf-8')
        count += 1
        # With probability 1/count, choose the current line.
        if random.randint(1, count) == 1:
            random_line = line_str
    return random_line

# Query the infini-gram API endpoint for a given 6-gram.
def query_infinigram(ngram, index="v4_rpj_llama_s4", retries=3):
    url = "https://api.infini-gram.io/"
    payload = {
        "index": index,
        "query_type": "count",
        "query": ngram,
    }
    for i in range(retries):
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                # If count is found, return it (count should be a nonnegative integer)
                if "count" in result:
                    return result["count"]
        except Exception as e:
            # In case of an exception, wait a bit and then retry.
            time.sleep(1)
    # Return 0 if all attempts failed.
    return 0

# Process one document: extract text, pick 100 random 6-grams, and count how many exist in the corpus.
def process_document(doc, index="v4_rpj_llama_s4"):
    text = doc.get("text", "")
    doc_id = doc.get("id", "Unknown")
    words = text.split()
    if len(words) < 6:
        # Not enough tokens to form a 6-gram
        return doc_id, 0

    possible_positions = len(words) - 6 + 1
    # If there are at least 100 possible 6-grams, sample without replacement; otherwise, sample with replacement.
    if possible_positions >= 100:
        start_indices = random.sample(range(possible_positions), 100)
    else:
        start_indices = [random.choice(range(possible_positions)) for _ in range(100)]

    match_count = 0
    for idx in start_indices:
        ngram_tokens = words[idx:idx + 6]
        ngram = " ".join(ngram_tokens)
        count = query_infinigram(ngram, index=index)
        # Consider this 6-gram a "match" if the API count is greater than zero.
        if count > 0:
            match_count += 1
    return doc_id, match_count

def main():
    parser = argparse.ArgumentParser(description="Infini-gram 6-gram matching script.")
    parser.add_argument("N", type=int, help="Number of random .jsonl files to process")
    parser.add_argument("s3_path", type=str, help="S3 path to a prefix containing .jsonl files (e.g., s3://my-bucket/my-prefix/)")
    parser.add_argument("--index", type=str, default="v4_dolma-v1_7_llama", help="Infini-gram index to use (default: v4_dolma-v1_7_llama)")
    args = parser.parse_args()

    # Parse the S3 path (assumes format: s3://bucket/prefix/)
    if not args.s3_path.startswith("s3://"):
        print("Error: s3_path must start with 's3://'")
        return
    path_without_scheme = args.s3_path[5:]
    parts = path_without_scheme.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    # List objects under the specified prefix
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    # Only keep .jsonl files
    files = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".jsonl")]
    if not files:
        print("No .jsonl files found in the given prefix.")
        return

    if args.N > len(files):
        print(f"Requested {args.N} files, but only found {len(files)}. Processing all available files.")
        args.N = len(files)
    # Randomly pick N files
    random_files = random.sample(files, args.N)

    total_matches = 0
    results = []
    for key in random_files:
        print(f"Processing file: {key}")
        line = get_random_line_from_s3(bucket, key)
        if not line:
            print(f"Skipping {key}: No valid lines found.")
            continue
        try:
            doc = json.loads(line)
        except Exception as e:
            print(f"Error parsing JSON in {key}: {e}")
            continue
        doc_id, match_count = process_document(doc, index=args.index)
        results.append((doc_id, match_count))
        total_matches += match_count

    # Output the breakdown
    for doc_id, match_count in results:
        print(f"Document ID: {doc_id} | Matched 6-grams: {match_count}")
    print(f"Total matched 6-grams: {total_matches}")

if __name__ == "__main__":
    main()
