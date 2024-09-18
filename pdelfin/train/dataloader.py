import json
from datasets import load_dataset, Dataset, Features, Value
import boto3
from typing import Dict, Any
import logging
import re
import random


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_s3_files(s3_path: str):
    """
    Lists files in the specified S3 path that match the glob pattern.
    """
    s3 = boto3.client('s3')
    match = re.match(r"s3://([^/]+)/(.+)", s3_path)
    if not match:
        logger.error(f"Invalid S3 path: {s3_path}")
        raise ValueError(f"Invalid S3 path: {s3_path}")
    
    bucket, prefix_pattern = match.groups()
    prefix = prefix_pattern.split('*')[0]  # Extract prefix before the wildcard
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    files = []
    pattern = re.compile(prefix_pattern.replace('*', '.*'))
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            if pattern.fullmatch(key):
                files.append(f"s3://{bucket}/{key}")
    return files


def load_jsonl_from_s3(s3_glob_path: str, first_n_files: int=None) -> Dataset:
    """
    Loads JSONL files from the specified S3 path into a Hugging Face Dataset.
    """
    all_s3_files = list_s3_files(s3_glob_path)

    if first_n_files:
        all_s3_files = all_s3_files[:first_n_files]
    
    # Use datasets library to load JSON files from S3
    dataset = load_dataset(
        'json',
        data_files=all_s3_files,
    )

    return dataset

def extract_openai_batch_query(query: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts necessary fields from a query entry.
    """
    custom_id = query.get('custom_id', '')
    body = query.get('body', {})
    messages = body.get('messages', [])
    
    input_prompt_text = ""
    input_prompt_image_base64 = ""
    
    for message in messages:
        if message.get('role') != 'user':
            continue  # We are only interested in user messages
        
        contents = message.get('content', [])
        for content_item in contents:
            if content_item.get('type') == 'text':
                input_prompt_text = content_item.get('text', "")
            elif content_item.get('type') == 'image_url':
                image_url = content_item.get('image_url', {}).get('url', "")
                if image_url.startswith('data:image'):
                    # Extract base64 part from data URL
                    try:
                        base64_data = image_url.split(',', 1)[1]
                        input_prompt_image_base64 = base64_data
                    except IndexError:
                        input_prompt_image_base64 = ""
    
    return {
        'custom_id': custom_id,
        'input_prompt_text': input_prompt_text,
        'input_prompt_image_base64': input_prompt_image_base64
    }

def build_batch_query_response_vision_dataset(query_glob_path: str, response_glob_path: str) -> Dataset:
    query_ds = load_jsonl_from_s3(query_glob_path)
    response_ds = load_jsonl_from_s3(response_glob_path)

    # Now merge them based on the custom_id field


    return query_ds
