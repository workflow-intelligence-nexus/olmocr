# Sends list of batch files to OpenAI for processing
import os
import time
import argparse
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up OpenAI client (API key should be set in the environment)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to upload a file to OpenAI and start batch processing
def upload_and_start_batch(file_path):
    try:
        # Upload the file to OpenAI
        with open(file_path, 'rb') as file:
            print(f"Uploading {file_path} to OpenAI Batch API...")
            upload_response = client.files.create(file=file, purpose="batch")
            file_id = upload_response.id
            print(f"File uploaded successfully: {file_id}")

        # Create a batch job
        print(f"Creating batch job for {file_path}...")
        batch_response = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": "pdf gold/silver data"
            }
        )
        
        batch_id = batch_response.id
        print(f"Batch created successfully: {batch_id}")
        return batch_id

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


# Main function to process all .jsonl files in a folder with multithreading
def process_folder(folder_path, batch_id_file, max_workers=8):
    # List all .jsonl files in the specified folder
    jsonl_files = [f for f in os.listdir(folder_path) if f.endswith('.jsonl')]

    if not jsonl_files:
        print("No .jsonl files found in the folder.")
        return

    batch_ids = []

    # Use ThreadPoolExecutor to process files concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a dictionary to store futures
        futures = {executor.submit(upload_and_start_batch, os.path.join(folder_path, jsonl_file)): jsonl_file for jsonl_file in jsonl_files}

        # Use tqdm to show progress and collect batch IDs as files are processed
        for future in tqdm(as_completed(futures), total=len(jsonl_files), desc="Processing files"):
            jsonl_file = futures[future]
            try:
                batch_id = future.result()
                if batch_id:
                    batch_ids.append(batch_id)
            except Exception as e:
                print(f"Error processing {jsonl_file}: {str(e)}")

    print(f"All files processed. Created {len(batch_ids)} batch jobs.")

    with open(batch_id_file, "w") as f:
        for id in batch_ids:
            f.write(id)
            f.write("\n")

    return batch_ids


if __name__ == "__main__":
    # Set up argument parsing for folder input
    parser = argparse.ArgumentParser(description='Upload .jsonl files and process batches in OpenAI API.')
    parser.add_argument('folder', type=str, help='Path to the folder containing .jsonl files')
    parser.add_argument('--batch_id_file', type=str, help="Path to a file where we store the batch ids to be retreived later")
    parser.add_argument('--max_workers', type=int, default=8, help='Number of files to process concurrently (default: 8)')
    args = parser.parse_args()

    # Process the folder and start batches
    process_folder(args.folder, args.batch_id_file, max_workers=args.max_workers)
