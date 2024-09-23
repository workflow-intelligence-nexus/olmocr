# Takes in a list of openai batch ids, and downloads the results to a folder
# Sends list of batch files to OpenAI for processing
import os
import time
import argparse
import openai
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up OpenAI client (API key should be set in the environment)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def download_batch_result(batch_id, output_folder):
    try:
        # Retrieve the batch result from OpenAI API
        batch_data = client.batches.retrieve(batch_id)

        if batch_data.status != "completed":
            return batch_id, False

        file_response = client.files.content(batch_data.output_file_id)

        # Define output file path
        output_file = os.path.join(output_folder, f"{batch_id}.json")

        # Save the result to a file
        with open(output_file, 'w') as f:
            f.write(str(file_response.text))
        
        return batch_id, True
    except Exception as e:
        print(e)
        return batch_id, False

if __name__ == "__main__":
    # Set up argument parsing for folder input
    parser = argparse.ArgumentParser(description='Retrieve the data from completed OpenAI Batch requests')
    parser.add_argument('--batch_id_file', type=str, required=True, help="Path to a file where we store the batch ids to be retrieved later")
    parser.add_argument('--output_folder', type=str, required=True, help="Save all the downloaded files there")
    args = parser.parse_args()

    # Ensure output folder exists
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Read the batch ids from the file
    with open(args.batch_id_file, 'r') as f:
        batch_ids = [line.strip() for line in f.readlines()]

    # Progress bar for batch downloads
    with tqdm(total=len(batch_ids), desc="Downloading batches", unit="batch") as pbar:
        # Use ThreadPoolExecutor to download in parallel (8 threads)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(download_batch_result, batch_id, args.output_folder) for batch_id in batch_ids]

            for future in as_completed(futures):
                batch_id, success = future.result()
                if success:
                    pbar.set_postfix({"Last batch": batch_id, "Status": "Success"})
                else:
                    pbar.set_postfix({"Last batch": batch_id, "Status": "Failed"})
                pbar.update(1)

    print("Download complete!")
