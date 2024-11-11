import logging
import argparse
import boto3
import signal
import os
import sys
import time
import subprocess
import hashlib
import json
import base64
import atexit
import asyncio
import aiohttp
import tempfile

from tqdm import tqdm
from io import BytesIO
from PIL import Image
from pypdf import PdfReader
from dataclasses import dataclass
from typing import Optional

from pdelfin.s3_utils import expand_s3_glob, get_s3_bytes, parse_s3_path, download_zstd_csv, upload_zstd_csv, download_directory
from pdelfin.data.renderpdf import render_pdf_to_base64png
from pdelfin.prompts import build_finetuning_prompt, PageResponse
from pdelfin.prompts.anchor import get_anchor_text
from pdelfin.check import check_poppler_version

# Basic logging setup for now
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

# Quiet logs from pypdf
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Global s3 client for the whole script, feel free to adjust params if you need it
workspace_s3 = boto3.client('s3')
pdf_s3 = boto3.client('s3')

MAX_TOKENS = 3000

@dataclass(frozen=True)
class PageResult:
    s3_path: str
    page_num: int
    response: PageResponse


async def build_page_query(local_pdf_path: str, page: int, target_longest_image_dim: int, target_anchor_text_len: int, image_rotation: int=0) -> dict:
    assert image_rotation in [0, 90, 180, 270], "Invalid image rotation provided in build_page_query"

    # Allow the page rendering to process in the background while we get the anchor text (which blocks the main thread)
    image_base64 = asyncio.to_thread(render_pdf_to_base64png, local_pdf_path, page, target_longest_image_dim=target_longest_image_dim)
    anchor_text = get_anchor_text(local_pdf_path, page, pdf_engine="pdfreport", target_length=target_anchor_text_len)

    image_base64 = await image_base64
    if image_rotation != 0:
        image_bytes = base64.b64decode(image_base64)
        with Image.open(BytesIO(image_bytes)) as img:
            rotated_img = img.rotate(-image_rotation, expand=True)

            # Save the rotated image to a bytes buffer
            buffered = BytesIO()
            rotated_img.save(buffered, format="PNG")

        # Encode the rotated image back to base64
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {
        "model": "Qwen/Qwen2-VL-7B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_finetuning_prompt(anchor_text)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ],
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.8
    }


def compute_workgroup_sha1(work_group: list[str]) -> str:
    sha1 = hashlib.sha1()
    # Ensure consistent ordering by sorting the list
    for pdf in sorted(work_group):
        sha1.update(pdf.encode('utf-8'))
    return sha1.hexdigest()


async def populate_pdf_work_queue(args):
    index_file_s3_path = os.path.join(args.workspace, "pdf_index_list.csv.zstd")

    if args.pdfs.startswith("s3://"):
        logger.info(f"Expanding s3 glob at {args.pdfs}")
        all_pdfs = expand_s3_glob(pdf_s3, args.pdfs)
    elif os.path.exists(args.pdfs):
        logger.info(f"Loading file at {args.pdfs}")
        with open(args.pdfs, "r") as f:
            all_pdfs = list(filter(None, (line.strip() for line in tqdm(f, desc="Processing PDFs"))))
    else:
        raise ValueError("pdfs argument needs to be either an s3 glob search path, or a local file contains pdf paths (one per line)")

    all_pdfs = set(all_pdfs)
    logger.info(f"Found {len(all_pdfs):,} total pdf paths")

    existing_lines = download_zstd_csv(workspace_s3, index_file_s3_path)

    # Parse existing work items into groups
    existing_groups = {}
    for line in existing_lines:
        if line.strip():
            parts = line.strip().split(",")
            group_hash = parts[0]
            group_pdfs = parts[1:]
            existing_groups[group_hash] = group_pdfs
    existing_pdf_set = set(pdf for group_pdfs in existing_groups.values() for pdf in group_pdfs)

    logger.info(f"Loaded {len(existing_pdf_set):,} existing pdf paths from the workspace")

    # Remove existing PDFs from all_pdfs
    new_pdfs = all_pdfs - existing_pdf_set
    logger.info(f"{len(new_pdfs):,} new pdf paths to add to the workspace")

    # Group the new PDFs into chunks of group_size
    # TODO: Figure out the group size automatically by sampling a few pdfs, and taking the mean/median number of pages, etc.
    new_groups = []
    current_group = []
    for pdf in sorted(new_pdfs):  # Sort for consistency
        current_group.append(pdf)
        if len(current_group) == args.group_size:
            group_hash = compute_workgroup_sha1(current_group)
            new_groups.append((group_hash, current_group))
            current_group = []
    if current_group:
        group_hash = compute_workgroup_sha1(current_group)
        new_groups.append((group_hash, current_group))

    logger.info(f"Created {len(new_groups):,} new work groups")

    # Combine existing groups with new groups
    combined_groups = existing_groups.copy()
    for group_hash, group_pdfs in new_groups:
        combined_groups[group_hash] = group_pdfs

    # Prepare lines to write back
    combined_lines = [",".join([group_hash] + group_pdfs) for group_hash, group_pdfs in combined_groups.items()]

    # Upload the combined work items back to S3
    if new_groups:
        upload_zstd_csv(workspace_s3, index_file_s3_path, combined_lines)

    logger.info("Completed adding new PDFs.")

async def load_pdf_work_queue(args) -> asyncio.Queue:
    index_file_s3_path = os.path.join(args.workspace, "pdf_index_list.csv.zstd")
    output_glob = f"{args.workspace}/dolma_documents/output_*.jsonl"

    # Define the two blocking I/O operations
    download_task = asyncio.to_thread(download_zstd_csv, workspace_s3, index_file_s3_path)
    expand_task = asyncio.to_thread(expand_s3_glob, workspace_s3, output_glob)

    # Run both tasks concurrently
    work_queue_lines, done_work_items = await asyncio.gather(download_task, expand_task)

    # Process the work queue lines
    work_queue = {
        parts[0]: parts[1:]
        for line in work_queue_lines
        if (parts := line.strip().split(",")) and line.strip()
    }

    # Extract done work hashes
    done_work_hashes = {
        os.path.basename(item)[len('output_'):-len('.jsonl')]
        for item in done_work_items
        if os.path.basename(item).startswith('output_') and os.path.basename(item).endswith('.jsonl')
    }

    # Determine remaining work
    remaining_work_hashes = set(work_queue) - done_work_hashes
    remaining_work_queue = {
        hash_: work_queue[hash_]
        for hash_ in remaining_work_hashes
    }

    # Populate the asyncio.Queue with remaining work
    queue = asyncio.Queue()
    for work, pdfs in remaining_work_queue.items():
        await queue.put((work, pdfs))

    return queue


async def process_page(args, session: aiohttp.ClientSession, pdf_s3_path: str, pdf_local_path: str, page_num: int) -> PageResult:
    COMPLETION_URL = "http://localhost:30000/v1/chat/completions"
    
    query = await build_page_query(
        pdf_path,
        page_num,
        args.target_longest_image_dim,
        args.target_anchor_text_len
    )
 
    try:
        async with session.post(COMPLETION_URL, json=query) as response:
            if response.status != 200:
                logger.warning(f"Request failed with status {response.status} for page {page_num}")
                return None
                
            try:
                base_response_data = await response.json()

                model_response_json = json.loads(base_response_data["choices"][0]["message"]["content"])
                page_response = PageResponse(**model_response_json)

                return PageResult(pdf_s3_path, page_num, page_response)
            except Exception as e:
                logger.warning(f"Could not parse response for {pdf_path}-{page_num}, reason: {e}")
            
            raise ValueError("Could not process page")
    except Exception as e:
        logger.error(f"Exception while processing page {page_num}: {e}")
        raise


async def process_pdf(args, pdf_s3_path: str):
    with tempfile.NamedTemporaryFile("wb+", suffix=".pdf") as tf:
        # TODO Switch to aioboto3 or something
        data = await asyncio.to_thread(lambda: get_s3_bytes(pdf_s3, pdf_s3_path))
        tf.write(data)
        tf.flush()

        reader = PdfReader(tf.name)
        num_pages = reader.get_num_pages()

        # List to hold the tasks for processing each page
        page_tasks = []

        async with aiohttp.ClientSession() as session:
            for page_num in range(1, num_pages + 1):
                # Create a task for each page
                task = asyncio.create_task(process_page(args, session, pdf_s3_path, tf.name, page_num))
                page_tasks.append(task)

            # Gather results from all page processing tasks
            try:
                page_results: list[PageResult] = await asyncio.gather(*page_tasks)
            except:
                logger.warning(f"Could not load page for {pdf_s3_path}, aborting document")
                return None

 
        # Build the document text and page spans
        document_text = ''
        pdf_page_spans = []
        current_char_pos = 0

        for index, page_result in enumerate(page_results):
            if page_result.response.natural_text is not None:
                content = page_result.response.natural_text + ("\n" if index == len(page_results) - 1 else "")
            else:
                content = ""

            start_pos = current_char_pos
            document_text += content
            current_char_pos = len(document_text)
            pdf_page_spans.append({
                'pdf_page_number': page_num,
                'start_char': start_pos,
                'end_char': current_char_pos
            })

        if not document_text:
            return None  # Return None if the document text is empty

        # Build the Dolma document
        metadata = {
            "Source-File": pdf_s3_path,
            "pdf-total-pages": num_pages,
        }

        id_ = hashlib.sha1(document_text.encode()).hexdigest()

        dolma_doc = {
            "id": id_,
            "text": document_text,
            "source": "pdelfin",
            "added": datetime.datetime.now().strftime("%Y-%m-%d"),
            "created": datetime.datetime.now().strftime("%Y-%m-%d"),
            "metadata": metadata,
            "attributes": {
                "pdf_page_numbers": pdf_page_spans
            }
        }

        return dolma_doc


async def worker(args, queue):
    while True:
        [work_hash, pdfs] = await queue.get()
        
        completed_pdfs = await asyncio.gather(*[process_pdf(args, pdf) for pdf in pdfs])
        
        # Take all the not None completed_pdfs and write them as a jsonl to the workspace output location
        # under the proper work_hash location
        for dolma_doc in completed_pdfs:
            logger.info("Done!", dolma_doc)
        
        queue.task_done()


async def sglang_server_task(args):
    model_cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'pdelfin', 'model')
    # TODO cache locally
    #download_directory(args.model, model_cache_dir)

    # Check the rope config and make sure it's got the proper key
    with open(os.path.join(model_cache_dir, "config.json"), "r") as cfin:
        config_data = json.load(cfin)

    if "rope_type" in config_data["rope_scaling"]:
        del config_data["rope_scaling"]["rope_type"]
        config_data["rope_scaling"]["type"] = "mrope"

        with open(os.path.join(model_cache_dir, "config.json"), "w") as cfout:
            json.dump(config_data, cfout)

    proc = await asyncio.create_subprocess_exec(
        "python3",
        
        "-m", "sglang.launch_server",
        "--model-path", model_cache_dir,
        "--chat-template", args.model_chat_template,
        "--context-length", str(args.model_max_context),
        )

    # Make really sure we kill this subprocess on exit
    def _kill_proc():
        proc.terminate()

    atexit.register(_kill_proc)

    await proc.wait()


async def sglang_server_ready():
    max_attempts = 300
    delay_sec = 1
    url = 'http://localhost:30000/v1/models'

    for attempt in range(1, max_attempts + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        logger.info("sglang server is ready.")
                        return
                    else:
                        logger.info(f"Attempt {attempt}: Unexpected status code {response.status}")
        except Exception as e:
            logger.warning(f"Attempt {attempt}: Exception occurred: {e}")

        await asyncio.sleep(delay_sec)

    raise Exception("sglang server did not become ready after waiting.")


async def main():
    parser = argparse.ArgumentParser(description='Manager for running millions of PDFs through a batch inference pipeline')
    parser.add_argument('workspace', help='The S3 path where work will be done e.g., s3://bucket/prefix/')
    parser.add_argument('--pdfs', help='Path to add pdfs stored in s3 to the workspace, can be a glob path s3://bucket/prefix/*.pdf or path to file containing list of pdf paths', default=None)
    parser.add_argument('--target_longest_image_dim', type=int, help='Dimension on longest side to use for rendering the pdf pages', default=1024)
    parser.add_argument('--target_anchor_text_len', type=int, help='Maximum amount of anchor text to use (characters)', default=6000)
    parser.add_argument('--workspace_profile', help='S3 configuration profile for accessing the workspace', default=None)
    parser.add_argument('--pdf_profile', help='S3 configuration profile for accessing the raw pdf documents', default=None)
    parser.add_argument('--group_size', type=int, default=20, help='Number of pdfs that will be part of each work item in the work queue.')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers to run at a time')

    parser.add_argument('--model', help='List of paths where you can find the model to convert this pdf. You can specify several different paths here, and the script will try to use the one which is fastest to access',
                         default=["weka://oe-data-default/jakep/Qwen_Qwen2-VL-7B-Instruct-e4ecf8-01JAH8GMWHTJ376S2N7ETXRXH4/best_bf16/",
                                  "gs://ai2-oe-data/jakep/experiments/qwen2vl-pdf/v1/models/jakep/Qwen_Qwen2-VL-7B-Instruct-e4ecf8-01JAH8GMWHTJ376S2N7ETXRXH4/checkpoint-9500/bf16/",
                                  "s3://ai2-oe-data/jakep/experiments/qwen2vl-pdf/v1/models/jakep/Qwen_Qwen2-VL-7B-Instruct-e4ecf8-01JAH8GMWHTJ376S2N7ETXRXH4/checkpoint-9500/bf16/"])
    parser.add_argument('--model_max_context', type=int, default="8192", help="Maximum context length that the model was fine tuned under")
    parser.add_argument('--model_chat_template', type=str, default="qwen2-vl", help="Chat template to pass to sglang server")
    args = parser.parse_args()

    if args.workspace_profile:
        global workspace_s3
        workspace_session = boto3.Session(profile_name=args.workspace_profile)
        workspace_s3 = workspace_session.client("s3")

    if args.pdf_profile:
        global pdf_s3
        pdf_session = boto3.Session(profile_name=args.pdf_profile)
        pdf_s3 = pdf_session.client("s3")

    check_poppler_version()
    logger.info(f"Starting pipeline with PID {os.getpid()}")

    if args.pdfs:
        await populate_pdf_work_queue(args)

    sglang_server = asyncio.create_task(sglang_server_task(args))

    work_queue = await load_pdf_work_queue(args)
    logger.info(f"Work queue prepared with {work_queue.qsize()} items")

    await sglang_server_ready()

    # Create worker tasks to process the queue concurrently.
    worker_tasks = []
    for i in range(args.workers):
        task = asyncio.create_task(worker(args, work_queue))
        worker_tasks.append(task)

    # Wait for the queue to be fully processed
    await work_queue.join()

    # Cancel our worker tasks.
    for task in worker_tasks:
        task.cancel()

    # Wait until all worker tasks are cancelled.
    await asyncio.gather(*worker_tasks, return_exceptions=True)
    
    # Wait for server to stop
    sglang_server.cancel()
    await sglang_server
    

if __name__ == "__main__":
    asyncio.run(main())

    # TODO
    # If there is a beaker flag, then your job is to trigger this script with N replicas on beaker
    # If not, then your job is to do the actual work

    # TODO
    # Possible future addon, in beaker, discover other nodes on this same job
    # Send them a message when you take a work item off the queue

