import os
import glob
import posixpath

from typing import Optional
from urllib.parse import urlparse


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