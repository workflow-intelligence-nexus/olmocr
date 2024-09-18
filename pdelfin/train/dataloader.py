import json
import logging
import multiprocessing
import re
from functools import partial
from typing import Any, Dict

import boto3
from datasets import Dataset, Features, Value, load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_s3_files(s3_path: str):
    """
    Lists files in the specified S3 path that match the glob pattern.
    """
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


def load_jsonl_from_s3(s3_glob_path: str, first_n_files: int = None) -> Dataset:
    """
    Loads JSONL files from the specified S3 path into a Hugging Face Dataset.
    """
    all_s3_files = list_s3_files(s3_glob_path)

    if first_n_files:
        all_s3_files = all_s3_files[:first_n_files]

    # Use datasets library to load JSON files from S3
    dataset = load_dataset(
        "json",
        data_files=all_s3_files,
    )

    return dataset


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

    return {
        "custom_id": custom_id,
        "input_prompt_text": input_prompt_text,
        "input_prompt_image_base64": input_prompt_image_base64,
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


def build_batch_query_response_vision_dataset(query_glob_path: str, response_glob_path: str) -> Dataset:
    logger.info("Loading query and response datasets")
    query_data = load_jsonl_from_s3(query_glob_path)
    response_data = load_jsonl_from_s3(response_glob_path)

    # Map the datasets down to the core fields that we're going to need to make them easier to process
    logger.info("Mapping query data")
    query_data = query_data["train"]
    query_data = query_data.map(extract_openai_batch_query, remove_columns=query_data.column_names)

    logger.info("Mapping response data")
    response_data = response_data["train"]
    response_data = response_data.map(extract_openai_batch_response, remove_columns=response_data.column_names)

    # What we're going to do, is build an in-memory map for the response data from custom_id to row
    # This will let us do quick lookups when we do a merge step, but it will not scale past a certain point
    logger.info("Building custom_id to row map")
    custom_id_to_response_row = {}
    for row_id, entry in enumerate(response_data):
        custom_id_to_response_row[entry["custom_id"]] = row_id

    logger.info("Running merge map")
    final_dataset = query_data.map(
        partial(merge_query_response, response_data=response_data, response_map=custom_id_to_response_row),
        num_proc=multiprocessing.cpu_count(),
    )
    final_dataset = final_dataset.filter(lambda x: x["finish_reason"] == "stop")

    return final_dataset
