import argparse
import logging
from functools import partial
import os
import boto3
from datasets import Dataset
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from pdelfin.train.dataloader import build_batch_query_response_vision_dataset


def save_dataset_in_parquet(dataset: Dataset, output_dir: str, rows_per_file: int = 10000, s3_endpoint_url: str = None):
    logger.info("Saving dataset in Parquet files")
    
    # Check if the output is an S3 path
    is_s3 = output_dir.startswith("s3://")
    if is_s3:
        s3_client = boto3.client('s3', endpoint_url=s3_endpoint_url) if s3_endpoint_url else boto3.client('s3')
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    total_rows = len(dataset)
    for start_idx in range(0, total_rows, rows_per_file):
        end_idx = min(start_idx + rows_per_file, total_rows)
        file_name = f"dataset_{start_idx}_{end_idx}.parquet"
        if is_s3:
            # Saving to S3
            bucket_name, key_prefix = parse_s3_path(output_dir)
            output_path = f"{key_prefix}/{file_name}"
            local_temp_file = f"/tmp/{file_name}"
            logger.info(f"Saving rows {start_idx} to {end_idx} locally at {local_temp_file}")
            dataset.select(range(start_idx, end_idx)).to_parquet(local_temp_file)
            try:
                logger.info(f"Uploading {local_temp_file} to s3://{bucket_name}/{output_path}")
                s3_client.upload_file(local_temp_file, bucket_name, output_path)
            except (NoCredentialsError, PartialCredentialsError) as e:
                logger.error(f"Failed to upload to S3: {e}")
                raise
            finally:
                os.remove(local_temp_file)
        else:
            # Saving locally
            output_path = os.path.join(output_dir, file_name)
            logger.info(f"Saving rows {start_idx} to {end_idx} in {output_path}")
            dataset.select(range(start_idx, end_idx)).to_parquet(output_path)

def parse_s3_path(s3_path: str):
    """Parses an S3 path into bucket and key prefix."""
    if not s3_path.startswith("s3://"):
        raise ValueError("S3 path must start with 's3://'")
    path = s3_path[5:]
    bucket_name, _, key_prefix = path.partition('/')
    return bucket_name, key_prefix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and save dataset as Parquet files.")
    parser.add_argument("--query_path", type=str, required=True, help="Path to the query dataset JSONL files.")
    parser.add_argument("--response_path", type=str, required=True, help="Path to the response dataset JSONL files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory or S3 path to save the output Parquet files.")
    parser.add_argument("--num_proc", type=int, default=32, help="Number of processes to use for data processing.")
    parser.add_argument("--s3_endpoint_url", type=str, default=None, help="Custom S3 endpoint URL, e.g., for S3-compatible storage.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Build the dataset
    final_dataset = build_batch_query_response_vision_dataset(
        query_glob_path=args.query_path,
        response_glob_path=args.response_path,
        num_proc=args.num_proc
    )

    # Save the dataset as Parquet files
    save_dataset_in_parquet(final_dataset, args.output_dir, s3_endpoint_url=args.s3_endpoint_url)

    logger.info("Dataset processing and saving completed.")