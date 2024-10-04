# Sends list of batch files to OpenAI for processing
# However, it also waits and gets the files when they are done, saves its state, and 
# allows you to submit more than the 100GB of file request limits that the openaiAPI has
import os
import time
import json
import datetime
import argparse
from enum import Enum
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up OpenAI client (API key should be set in the environment)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_OPENAI_DISK_SPACE = 100 * 1024 * 1024 * 1024 # Max is 100GB on openAI
UPLOAD_STATE_FILENAME = "SENDSILVER_DATA"

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

def download_batch_result(batch_id, output_folder):
    # Retrieve the batch result from OpenAI API
    batch_data = client.batches.retrieve(batch_id)

    if batch_data.status != "completed":
        print(f"WARNING: {batch_id} is not completed, status: {batch_data.status}")
        return batch_id, False

    file_response = client.files.content(batch_data.output_file_id)

    # Define output file path
    output_file = os.path.join(output_folder, f"{batch_id}.json")

    # Save the result to a file
    with open(output_file, 'w') as f:
        f.write(str(file_response.text))

    return batch_id, True

    

ALL_STATES = ["init", "processing", "completed", "errored_out", "could_not_upload"]
FINISHED_STATES = [ "completed", "errored_out" ]

def _json_datetime_decoder(obj):
    if 'last_checked' in obj:
        try:
            obj['last_checked'] = datetime.datetime.fromisoformat(obj['last_checked'])
        except (TypeError, ValueError):
            pass  # If it's not a valid ISO format, leave it as is
    return obj

def get_state(folder_path: str) -> dict:
    state_file = os.path.join(folder_path, UPLOAD_STATE_FILENAME)

    try:
        with open(state_file, "r") as f:
            return json.load(f, object_hook=_json_datetime_decoder)
    except (json.decoder.JSONDecodeError, FileNotFoundError):
        # List all .jsonl files in the specified folder
        jsonl_files = [f for f in os.listdir(folder_path) if f.endswith('.jsonl')]

        if not jsonl_files:
            raise Exception("No JSONL files found to process")
        
        state = {f: 
                    {
                        "filename": f,
                        "batch_id": None,
                        "state": "init",
                        "size": os.path.getsize(os.path.join(folder_path, f)),
                        "last_checked": datetime.datetime.now().isoformat(),
                    } for f in jsonl_files}

        with open(state_file, "w") as f:
            return json.dump(state, f)
        
        return state

def update_state(folder_path: str, filename: str, **kwargs):
    all_state = get_state(folder_path)
    for kwarg_name, kwarg_value in kwargs.items():
        all_state[filename][kwarg_name] = kwarg_value

    all_state[filename]["last_checked"] = datetime.datetime.now().isoformat()

    state_file = os.path.join(folder_path, UPLOAD_STATE_FILENAME)
    with open(state_file, "w") as f:
        return json.dump(all_state, f)
        
def get_total_space_usage():
    return sum(file.size for file in client.files.list())

def get_estimated_space_usage(folder_path):
    all_states = get_state(folder_path)
    return sum(s["size"] for s in all_states.values() if s["state"] == "processing")

def get_next_work_item(folder_path):
    all_states = get_state(folder_path)
    all_states = [s for s in all_states.values() if s["state"] not in FINISHED_STATES]
    all_states.sort(key=lambda s: s["last_checked"])

    return all_states[0] if len(all_states) > 0 else None



# Main function to process all .jsonl files in a folder
def process_folder(folder_path: str, max_gb: int):
    output_folder = f"{folder_path}_done"
    os.makedirs(output_folder, exist_ok=True)

    starting_free_space = MAX_OPENAI_DISK_SPACE - get_total_space_usage()

    if starting_free_space < max_gb * 2:
        raise ValueError(f"Insufficient free space in OpenAI's file storage: Only {starting_free_space} GB left, but 2x{max_gb} GB are required (1x for your uploads, 1x for your results).")

    while not all(state["state"] in FINISHED_STATES for state in get_state(folder_path).values()):
        work_item = get_next_work_item(folder_path)
        print(f"Processing {os.path.basename(work_item['filename'])}, cur status = {work_item['state']}")

        # If all work items have been checked on, then you need to sleep a bit
        if work_item["last_checked"] > datetime.datetime.now() - datetime.timedelta(seconds=1):
            time.sleep(1)

        if work_item["state"] == "init":
            if starting_free_space - get_estimated_space_usage(folder_path) > 0:
                try:
                    batch_id = upload_and_start_batch(work_item["filename"])
                    update_state(folder_path, work_item["filename"], state="processing", batch_id=batch_id)
                except:
                    update_state(folder_path, work_item["filename"], state="init")
            else:
                print("waiting for something to finish processing before uploading more")
        elif work_item["state"] == "processing":
            batch_data = client.batches.retrieve(work_item["batch_id"])

            if batch_data.status == "completed":
                batch_id, success = download_batch_result(work_item["batch_id"], output_folder)

                if success:
                    update_state(folder_path, work_item["filename"], state="completed")
                else:
                    update_state(folder_path, work_item["filename"], state="errored_out")

                client.files.delete(batch_data.input_file_id)
                client.files.delete(batch_data.output_file_id)
            elif batch_data.status in ["failed", "expired", "cancelled"]:
                update_state(folder_path, work_item["filename"], state="errored_out")

                try:
                    client.files.delete(batch_data.input_file_id)
                except:
                    print("Could not delete old file data")


if __name__ == "__main__":
    # Set up argument parsing for folder input
    parser = argparse.ArgumentParser(description='Upload .jsonl files and process batches in OpenAI API.')
    parser.add_argument("--max_gb", type=int, default=25, help="Max number of GB of batch processing files to upload at one time")
    parser.add_argument('folder', type=str, help='Path to the folder containing .jsonl files')
    
    args = parser.parse_args()

    # Process the folder and start batches
    process_folder(args.folder, args.max_gb)
