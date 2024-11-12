import os
import glob
import posixpath
import logging
import tempfile
import boto3
import requests
import concurrent.futures
import hashlib  # Added for MD5 hash computation

from urllib.parse import urlparse
from pathlib import Path
from google.auth import compute_engine
from google.cloud import storage
from botocore.config import Config
from botocore.exceptions import NoCredentialsError
from boto3.s3.transfer import TransferConfig
from typing import Optional, List
from urllib.parse import urlparse
import zstandard as zstd
from io import BytesIO, TextIOWrapper
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_s3_path(s3_path: str) -> tuple[str, str]:
    if not (s3_path.startswith('s3://') or s3_path.startswith('gs://') or s3_path.startswith('weka://')):
        raise ValueError('s3_path must start with s3://, gs://, or weka://')
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

def download_directory(model_choices: List[str], local_dir: str):
    """
    Download the model to a specified local directory.
    The function will attempt to download from the first available source in the provided list.
    Supports Weka (weka://), Google Cloud Storage (gs://), and Amazon S3 (s3://) links.

    Args:
        model_choices (List[str]): List of model paths (weka://, gs://, or s3://).
        local_dir (str): Local directory path where the model will be downloaded.

    Raises:
        ValueError: If no valid model path is found in the provided choices.
    """
    local_path = Path(os.path.expanduser(local_dir))
    local_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Local directory set to: {local_path}")

    # Reorder model_choices to prioritize weka:// links
    weka_choices = [path for path in model_choices if path.startswith("weka://")]
    other_choices = [path for path in model_choices if not path.startswith("weka://")]
    prioritized_choices = weka_choices + other_choices

    for model_path in prioritized_choices:
        logger.info(f"Attempting to download from: {model_path}")
        try:
            if model_path.startswith("weka://"):
                download_dir_from_storage(
                    model_path, str(local_path), storage_type='weka')
                logger.info(f"Successfully downloaded model from Weka: {model_path}")
                return
            elif model_path.startswith("gs://"):
                download_dir_from_storage(
                    model_path, str(local_path), storage_type='gcs')
                logger.info(f"Successfully downloaded model from Google Cloud Storage: {model_path}")
                return
            elif model_path.startswith("s3://"):
                download_dir_from_storage(
                    model_path, str(local_path), storage_type='s3')
                logger.info(f"Successfully downloaded model from S3: {model_path}")
                return
            else:
                logger.warning(f"Unsupported model path scheme: {model_path}")
        except Exception as e:
            logger.error(f"Failed to download from {model_path}: {e}")
            continue

    raise ValueError("Failed to download the model from all provided sources.")


def download_dir_from_storage(storage_path: str, local_dir: str, storage_type: str):
    """
    Generalized function to download model files from different storage services
    to a local directory, syncing using MD5 hashes where possible.

    Args:
        storage_path (str): The path to the storage location (weka://, gs://, or s3://).
        local_dir (str): The local directory where files will be downloaded.
        storage_type (str): Type of storage ('weka', 'gcs', or 's3').

    Raises:
        ValueError: If the storage type is unsupported or credentials are missing.
    """
    bucket_name, prefix = parse_s3_path(storage_path)
    total_files = 0
    objects = []

    if storage_type == 'gcs':
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))
        total_files = len(blobs)
        logger.info(f"Found {total_files} files in GCS bucket '{bucket_name}' with prefix '{prefix}'.")

        def should_download(blob, local_file_path):
            return compare_hashes_gcs(blob, local_file_path)

        def download_blob(blob, local_file_path):
            blob.download_to_filename(local_file_path)

        items = blobs
    elif storage_type in ('s3', 'weka'):
        if storage_type == 'weka':
            weka_access_key = os.getenv("WEKA_ACCESS_KEY_ID")
            weka_secret_key = os.getenv("WEKA_SECRET_ACCESS_KEY")
            if not weka_access_key or not weka_secret_key:
                raise ValueError("WEKA_ACCESS_KEY_ID and WEKA_SECRET_ACCESS_KEY must be set for Weka access.")
            endpoint_url = "https://weka-aus.beaker.org:9000"
            boto3_config = Config(
                max_pool_connections=500,
                signature_version='s3v4',
                retries={'max_attempts': 10, 'mode': 'standard'}
            )
            s3_client = boto3.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=weka_access_key,
                aws_secret_access_key=weka_secret_key,
                config=boto3_config
            )
        else:
            s3_client = boto3.client('s3', config=Config(max_pool_connections=500))

        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        for page in pages:
            if 'Contents' in page:
                objects.extend(page['Contents'])
        total_files = len(objects)
        logger.info(f"Found {total_files} files in {'Weka' if storage_type == 'weka' else 'S3'} bucket '{bucket_name}' with prefix '{prefix}'.")

        transfer_config = TransferConfig(
            multipart_threshold=8 * 1024 * 1024,
            multipart_chunksize=8 * 1024 * 1024,
            max_concurrency=100,
            use_threads=True
        )

        def should_download(obj, local_file_path):
            return compare_hashes_s3(obj, local_file_path)

        def download_blob(obj, local_file_path):
            s3_client.download_file(bucket_name, obj['Key'], local_file_path, Config=transfer_config)

        items = objects
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for item in items:
            if storage_type == 'gcs':
                relative_path = os.path.relpath(item.name, prefix)
            else:
                relative_path = os.path.relpath(item['Key'], prefix)
            local_file_path = os.path.join(local_dir, relative_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            if should_download(item, local_file_path):
                futures.append(executor.submit(download_blob, item, local_file_path))
            else:
                total_files -= 1  # Decrement total_files as we're skipping this file

        if total_files > 0:
            for _ in tqdm(concurrent.futures.as_completed(futures), total=total_files, desc=f"Downloading from {storage_type.upper()}"):
                pass
        else:
            logger.info("All files are up-to-date. No downloads needed.")

    logger.info(f"Downloaded model from {storage_type.upper()} to {local_dir}")


def compare_hashes_gcs(blob, local_file_path: str) -> bool:
    """Compare MD5 hashes for GCS blobs."""
    if os.path.exists(local_file_path):
        remote_md5_base64 = blob.md5_hash
        hash_md5 = hashlib.md5()
        with open(local_file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        local_md5 = hash_md5.digest()
        remote_md5 = base64.b64decode(remote_md5_base64)
        if remote_md5 == local_md5:
            logger.info(f"File '{local_file_path}' already up-to-date. Skipping download.")
            return False
        else:
            logger.info(f"File '{local_file_path}' differs from GCS. Downloading.")
            return True
    else:
        logger.info(f"File '{local_file_path}' does not exist locally. Downloading.")
        return True


def compare_hashes_s3(obj, local_file_path: str) -> bool:
    """Compare MD5 hashes or sizes for S3 objects (including Weka)."""
    if os.path.exists(local_file_path):
        etag = obj['ETag'].strip('"')
        if '-' in etag:
            remote_size = obj['Size']
            local_size = os.path.getsize(local_file_path)
            if remote_size == local_size:
                logger.info(f"File '{local_file_path}' size matches remote multipart file. Skipping download.")
                return False
            else:
                logger.info(f"File '{local_file_path}' size differs from remote multipart file. Downloading.")
                return True
        else:
            hash_md5 = hashlib.md5()
            with open(local_file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
            local_md5 = hash_md5.hexdigest()
            if etag == local_md5:
                logger.info(f"File '{local_file_path}' already up-to-date. Skipping download.")
                return False
            else:
                logger.info(f"File '{local_file_path}' differs from remote. Downloading.")
                return True
    else:
        logger.info(f"File '{local_file_path}' does not exist locally. Downloading.")
        return True