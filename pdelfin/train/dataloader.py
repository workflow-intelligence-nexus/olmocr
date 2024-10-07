import json
import logging
import multiprocessing
import re
import random
import base64
import glob

from functools import partial
from typing import Any, Dict, Optional
from logging import Logger

import boto3
from datasets import Dataset, Features, Value, load_dataset, concatenate_datasets, DatasetDict
from .core.config import DataConfig, SourceConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_jsonl_files(s3_path: str):
    """
    Lists files in the specified S3 path that match the glob pattern.
    """
    if s3_path.startswith("s3://"):
        s3 = boto3.client("s3")
        match = re.match(r"s3://([^/]+)/(.+)", s3_path)
        if not match:
            logger.error(f"Invalid S3 path: {s3_path}")
            raise ValueError(f"Invalid S3 path: {s3_path}")

        bucket, prefix_pattern = match.groups()
        prefix = prefix_pattern.split("*")[0]  # Extract prefix before the wildcard
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        files = []
        pattern = re.compile(prefix_pattern.replace("*", ".*"))
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if pattern.fullmatch(key):
                    files.append(f"s3://{bucket}/{key}")
        return files
    else:
        return glob.glob(s3_path)


def load_jsonl_into_ds(s3_glob_path: str, first_n_files: int = None) -> Dataset:
    """
    Loads JSONL files from the specified S3 path into a Hugging Face Dataset.
    """
    all_json_files = list_jsonl_files(s3_glob_path)

    if first_n_files:
        all_json_files = all_json_files[:first_n_files]

    # Use datasets library to load JSON files from S3
    dataset = load_dataset(
        "json",
        data_files=all_json_files,
    )

    return dataset


def get_png_dimensions_from_base64(base64_data) -> tuple[int, int]:
    """
    Returns the (width, height) of a PNG image given its base64-encoded data,
    without base64-decoding the entire data or loading the PNG itself

    Should be really fast to support filtering

    Parameters:
    - base64_data (str): Base64-encoded PNG image data.

    Returns:
    - tuple: (width, height) of the image.

    Raises:
    - ValueError: If the data is not a valid PNG image or the required bytes are not found.
    """
    # PNG signature is 8 bytes
    png_signature_base64 = base64.b64encode(b'\x89PNG\r\n\x1a\n').decode('ascii')
    if not base64_data.startswith(png_signature_base64[:8]):
        raise ValueError('Not a valid PNG file')

    # Positions in the binary data where width and height are stored
    width_start = 16  # Byte position where width starts (0-based indexing)
    width_end = 20    # Byte position where width ends (exclusive)
    height_start = 20
    height_end = 24

    # Compute the byte range needed (from width_start to height_end)
    start_byte = width_start
    end_byte = height_end

    # Calculate base64 character positions
    # Each group of 3 bytes corresponds to 4 base64 characters
    base64_start = (start_byte // 3) * 4
    base64_end = ((end_byte + 2) // 3) * 4  # Add 2 to ensure we cover partial groups

    # Extract the necessary base64 substring
    base64_substring = base64_data[base64_start:base64_end]

    # Decode only the necessary bytes
    decoded_bytes = base64.b64decode(base64_substring)

    # Compute the offset within the decoded bytes
    offset = start_byte % 3

    # Extract width and height bytes
    width_bytes = decoded_bytes[offset:offset+4]
    height_bytes = decoded_bytes[offset+4:offset+8]

    if len(width_bytes) < 4 or len(height_bytes) < 4:
        raise ValueError('Insufficient data to extract dimensions')

    # Convert bytes to integers
    width = int.from_bytes(width_bytes, 'big')
    height = int.from_bytes(height_bytes, 'big')

    return width, height


def extract_openai_batch_query(query: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts necessary fields from a query entry passed to openai's batch API for vision LMs
    """
    custom_id = query.get("custom_id", "")
    body = query.get("body", {})
    messages = body.get("messages", [])

    input_prompt_text = ""
    input_prompt_image_base64 = ""

    for message in messages:
        if message.get("role") != "user":
            continue  # We are only interested in user messages

        contents = message.get("content", [])
        for content_item in contents:
            if content_item.get("type") == "text":
                input_prompt_text = content_item.get("text", "")
            elif content_item.get("type") == "image_url":
                image_url = content_item.get("image_url", {}).get("url", "")
                if image_url.startswith("data:image"):
                    # Extract base64 part from data URL
                    try:
                        base64_data = image_url.split(",", 1)[1]
                        input_prompt_image_base64 = base64_data
                    except IndexError:
                        input_prompt_image_base64 = ""

    # At this point, the input_prompt_text is the raw text that was passed to the OpenAI model
    # to generate our silver data. But, we want to have a simplfied prompt for this here fine tune,
    # so we're going to extract out just the raw extracted prompt text
    pattern = r"RAW_TEXT_START\s*\n(.*?)\nRAW_TEXT_END"

    # Use re.DOTALL to ensure that the dot matches newline characters
    match = re.search(pattern, input_prompt_text, re.DOTALL)

    if match:
        raw_page_text = match.group(1).strip()
    else:
        raw_page_text = ""

    return {
        "custom_id": custom_id,
        "input_prompt_text": input_prompt_text,
        "input_prompt_image_base64": input_prompt_image_base64,
        "raw_page_text": raw_page_text,
    }


def extract_openai_batch_response(example):
    custom_id = example.get("custom_id", None)
    response_body = example.get("response", {}).get("body", {})
    choices = response_body.get("choices", [])
    response = ""
    finish_reason = ""
    if choices:
        first_choice = choices[0]
        message = first_choice.get("message", {})
        response = message.get("content", "")
        finish_reason = first_choice.get("finish_reason", "")

    return {"custom_id": custom_id, "response": response, "finish_reason": finish_reason}


def merge_query_response(query_example, response_data: Dataset, response_map: dict[str, int]):
    custom_id = query_example["custom_id"]

    if custom_id not in response_map:
        return {
            "response": None,
            "finish_reason": None,
        }

    response_row = response_data[response_map[custom_id]]

    return {"response": response_row["response"], "finish_reason": response_row["finish_reason"]}


def build_batch_query_response_vision_dataset(query_glob_path: str, response_glob_path: str, num_proc: int=32) -> Dataset:
    logger.info("Loading query and response datasets")
    query_data = load_jsonl_into_ds(query_glob_path)
    response_data = load_jsonl_into_ds(response_glob_path)

    # Map the datasets down to the core fields that we're going to need to make them easier to process
    logger.info("Mapping query data")
    query_data = query_data["train"]
    query_data = query_data.map(extract_openai_batch_query, remove_columns=query_data.column_names, num_proc=num_proc)

    logger.info("Mapping response data")
    response_data = response_data["train"]
    response_data = response_data.map(extract_openai_batch_response, remove_columns=response_data.column_names, num_proc=num_proc)

    # What we're going to do, is build an in-memory map for the response data from custom_id to row
    # This will let us do quick lookups when we do a merge step, but it will not scale past a certain point
    logger.info("Building custom_id to row map")
    custom_id_to_response_row = {}
    for row_id, entry in enumerate(response_data):
        custom_id_to_response_row[entry["custom_id"]] = row_id

    logger.info("Running merge map")
    final_dataset = query_data.map(
        partial(merge_query_response, response_data=response_data, response_map=custom_id_to_response_row),
        num_proc=num_proc
    )

    # Don't include data where the model cut off due to a length issue, or moderation issue
    final_dataset = final_dataset.filter(lambda x: x["finish_reason"] == "stop", num_proc=num_proc)

    # Pick things that have a reasonable image size only
    def pick_image_sizes(x):
        width, height = get_png_dimensions_from_base64(x["input_prompt_image_base64"])
        return 1800 <= max(width, height) <= 2200

    final_dataset = final_dataset.filter(pick_image_sizes, num_proc=num_proc)

    return final_dataset


def make_dataset(
    train_data_config: DataConfig,
    valid_data_config: Optional[DataConfig] = None,
    test_data_config: Optional[DataConfig] = None,
    num_proc: int = 32,
    logger: Optional[Logger] = None,
):
    logger = logger or get_logger(__name__)
    random.seed(train_data_config.seed)

    dataset_splits: Dict[str, Dataset] = {}
    tmp_train_sets = []

    logger.info("Loading training data from %s sources", len(train_data_config.sources))
    for source in train_data_config.sources:
        tmp_train_sets.append(
            build_batch_query_response_vision_dataset(source.query_glob_path, source.response_glob_path)
        )
    dataset_splits["train"] = concatenate_datasets(tmp_train_sets)
    logger.info(
        f"Loaded {len(dataset_splits['train'])} training samples from {len(train_data_config.sources)} sources"
    )

    if valid_data_config:
        tmp_validation_sets = []
        logger.info("Loading validation data from %s sources", len(valid_data_config.sources))
        for source in valid_data_config.sources:
            tmp_validation_sets.append(
                build_batch_query_response_vision_dataset(source.query_glob_path, source.response_glob_path)
            )
        dataset_splits["validation"] = concatenate_datasets(tmp_validation_sets)
        logger.info(
            f"Loaded {len(dataset_splits['validation'])} validation samples from {len(valid_data_config.sources)} sources"
        )

    if test_data_config:
        tmp_test_sets = []
        logger.info("Loading test data from %s sources", len(test_data_config.sources))
        for source in test_data_config.sources:
            tmp_test_sets.append(
                build_batch_query_response_vision_dataset(source.query_glob_path, source.response_glob_path)
            )
        dataset_splits["test"] = concatenate_datasets(tmp_test_sets)
        logger.info(
            f"Loaded {len(dataset_splits['test'])} test samples from {len(test_data_config.sources)} sources"
        )

    return DatasetDict(**dataset_splits)