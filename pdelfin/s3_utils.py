import os
import glob
import posixpath
import logging
import tempfile
import boto3
import requests
import concurrent.futures

from urllib.parse import urlparse
from pathlib import Path
from google.auth import compute_engine
from google.cloud import storage
from botocore.config import Config
from botocore.exceptions import NoCredentialsError
from typing import Optional
from urllib.parse import urlparse
import zstandard as zstd
from io import BytesIO, TextIOWrapper
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_s3_path(s3_path: str) -> tuple[str, str]:
    if not s3_path.startswith('s3://'):
        raise ValueError('s3_path must start with s3://')
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')

    return bucket, key


def expand_s3_glob(s3_client, s3_glob: str) -> dict[str, str]:
    parsed = urlparse(s3_glob)
    bucket_name = parsed.netloc
    prefix = os.path.dirname(parsed.path.lstrip('/')).rstrip('/') + "/"
    pattern = os.path.basename(parsed.path)

    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    matched_files = {}
    for page in page_iterator:
        for obj in page.get('Contents', []):
            key = obj['Key']
            if glob.fnmatch.fnmatch(key, posixpath.join(prefix, pattern)):
                matched_files[f"s3://{bucket_name}/{key}"] = obj['ETag'].strip('"')

    return matched_files


def get_s3_bytes(s3_client, s3_path: str, start_index: Optional[int] = None, end_index: Optional[int] = None) -> bytes:
    bucket, key = parse_s3_path(s3_path)

    # Build the range header if start_index and/or end_index are specified
    range_header = None
    if start_index is not None and end_index is not None:
        # Range: bytes=start_index-end_index
        range_value = f"bytes={start_index}-{end_index}"
        range_header = {'Range': range_value}
    elif start_index is not None and end_index is None:
        # Range: bytes=start_index-
        range_value = f"bytes={start_index}-"
        range_header = {'Range': range_value}
    elif start_index is None and end_index is not None:
        # Range: bytes=-end_index (last end_index bytes)
        range_value = f"bytes=-{end_index}"
        range_header = {'Range': range_value}

    if range_header:
        obj = s3_client.get_object(Bucket=bucket, Key=key, Range=range_header['Range'])
    else:
        obj = s3_client.get_object(Bucket=bucket, Key=key)

    return obj['Body'].read()


def put_s3_bytes(s3_client, s3_path: str, data: bytes):
    bucket, key = parse_s3_path(s3_path)

    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType='text/plain; charset=utf-8'
    )


def parse_custom_id(custom_id: str) -> tuple[str, int]:
    s3_path = custom_id[:custom_id.rindex("-")]
    page_num = int(custom_id[custom_id.rindex("-") + 1:])
    return s3_path, page_num


def download_zstd_csv(s3_client, s3_path):
    """Download and decompress a .zstd CSV file from S3."""
    try:
        compressed_data = get_s3_bytes(s3_client, s3_path)
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(compressed_data)
        text_stream = TextIOWrapper(BytesIO(decompressed), encoding='utf-8')
        lines = text_stream.readlines()
        logger.info(f"Downloaded and decompressed {s3_path}")
        return lines
    except s3_client.exceptions.NoSuchKey:
        logger.info(f"No existing {s3_path} found in s3, starting fresh.")
        return []


def upload_zstd_csv(s3_client, s3_path, lines):
    """Compress and upload a list of lines as a .zstd CSV file to S3."""
    joined_text = "\n".join(lines)
    compressor = zstd.ZstdCompressor()
    compressed = compressor.compress(joined_text.encode('utf-8'))
    put_s3_bytes(s3_client, s3_path, compressed)
    logger.info(f"Uploaded compressed {s3_path}")


def is_running_on_gcp():
    """Check if the script is running on a Google Cloud Platform (GCP) instance."""
    try:
        # GCP metadata server URL to check instance information
        response = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/",
            headers={"Metadata-Flavor": "Google"},
            timeout=1  # Set a short timeout
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def download_directory(model_choices: list[str], local_dir: str):
    """
    Download the model to a specified local directory.
    The function will attempt to download from the first available source in the provided list.
    Supports Google Cloud Storage (gs://) and Amazon S3 (s3://) links.

    Args:
        model_choices (list[str]): List of model paths (gs:// or s3://).
        local_dir (str): Local directory path where the model will be downloaded.

    Raises:
        ValueError: If no valid model path is found in the provided choices.
    """
    # Ensure the local directory exists
    local_path = Path(os.path.expanduser(local_dir))
    local_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Local directory set to: {local_path}")

    # Iterate through the provided choices and attempt to download from the first available source
    for model_path in model_choices:
        logger.info(f"Attempting to download from: {model_path}")
        try:
            if model_path.startswith("gs://"):
                download_dir_from_gcs(model_path, str(local_path))
                logger.info(f"Successfully downloaded model from Google Cloud Storage: {model_path}")
                return
            elif model_path.startswith("s3://"):
                download_dir_from_s3(model_path, str(local_path))
                logger.info(f"Successfully downloaded model from S3: {model_path}")
                return
            else:
                logger.warning(f"Unsupported model path scheme: {model_path}")
        except Exception as e:
            logger.error(f"Failed to download from {model_path}: {e}")
            continue  # Try the next available source

    raise ValueError("Failed to download the model from all provided sources.")


def download_dir_from_gcs(gcs_path: str, local_dir: str):
    """Download model files from Google Cloud Storage to a local directory."""
    client = storage.Client()
    bucket_name, prefix = parse_s3_path(gcs_path.replace("gs://", "s3://"))
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=prefix))
    total_files = len(blobs)
    logger.info(f"Found {total_files} files in GCS bucket '{bucket_name}' with prefix '{prefix}'.")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for blob in blobs:
            relative_path = os.path.relpath(blob.name, prefix)
            local_file_path = os.path.join(local_dir, relative_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            futures.append(executor.submit(blob.download_to_filename, local_file_path))

        # Use tqdm to display progress
        for _ in tqdm(concurrent.futures.as_completed(futures), total=total_files, desc="Downloading from GCS"):
            pass

    logger.info(f"Downloaded model from Google Cloud Storage to {local_dir}")


def download_dir_from_s3(s3_path: str, local_dir: str):
    """Download model files from S3 to a local directory."""
    boto3_config = Config(
        max_pool_connections=50  # Adjust this number based on your requirements
    )
    s3_client = boto3.client('s3', config=boto3_config)
    bucket, prefix = parse_s3_path(s3_path)
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    objects = []
    for page in pages:
        if 'Contents' in page:
            objects.extend(page['Contents'])

    total_files = len(objects)
    logger.info(f"Found {total_files} files in S3 bucket '{bucket}' with prefix '{prefix}'.")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for obj in objects:
            key = obj["Key"]
            relative_path = os.path.relpath(key, prefix)
            local_file_path = os.path.join(local_dir, relative_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            futures.append(executor.submit(s3_client.download_file, bucket, key, local_file_path))

        # Use tqdm to display progress
        for _ in tqdm(concurrent.futures.as_completed(futures), total=total_files, desc="Downloading from S3"):
            pass

    logger.info(f"Downloaded model from S3 to {local_dir}")
