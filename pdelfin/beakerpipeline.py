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
import datetime
import tempfile
import re

from tqdm import tqdm
from io import BytesIO
from PIL import Image
from pypdf import PdfReader
from functools import partial
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ProcessPoolExecutor

from pdelfin.s3_utils import expand_s3_glob, get_s3_bytes, parse_s3_path, download_zstd_csv, upload_zstd_csv, download_directory
from pdelfin.data.renderpdf import render_pdf_to_base64png
from pdelfin.prompts import build_finetuning_prompt, PageResponse
from pdelfin.prompts.anchor import get_anchor_text
from pdelfin.check import check_poppler_version
from pdelfin.metrics import MetricsKeeper, WorkerTracker

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

sglang_logger = logging.getLogger("sglang")
sglang_logger.propagate = False

file_handler = logging.FileHandler('beakerpipeline-debug.log', mode='a')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
sglang_logger.addHandler(file_handler)

# Quiet logs from pypdf
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Global s3 clients fo the whole script, we have two separate ones in case your workspace and your pdfs are in different accounts
workspace_s3 = boto3.client('s3')
pdf_s3 = boto3.client('s3')

# Global variables for token statistics
metrics = MetricsKeeper(window=60*5)
tracker = WorkerTracker()

# Process pool for offloading cpu bound work, like calculating anchor texts
process_pool = ProcessPoolExecutor()

@dataclass(frozen=True)
class PageResult:
    s3_path: str
    page_num: int
    response: PageResponse

    input_tokens: int
    output_tokens: int


async def build_page_query(local_pdf_path: str, page: int, target_longest_image_dim: int, target_anchor_text_len: int, image_rotation: int=0) -> dict:
    MAX_TOKENS = 3000
    assert image_rotation in [0, 90, 180, 270], "Invalid image rotation provided in build_page_query"

    # Allow the page rendering to process in the background while we get the anchor text (which blocks the main thread)
    image_base64 = asyncio.to_thread(render_pdf_to_base64png, local_pdf_path, page, target_longest_image_dim=target_longest_image_dim)

    # GET ANCHOR TEXT IS NOT THREAD SAFE!! Ahhhh..... don't try to do it
    # and it's also CPU bound, so it needs to run in a process pool
    loop = asyncio.get_running_loop()
    anchor_text = loop.run_in_executor(process_pool, partial(get_anchor_text, pdf_engine="pdfreport", target_length=target_anchor_text_len), local_pdf_path, page)

    image_base64, anchor_text = await asyncio.gather(image_base64, anchor_text)
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


async def process_page(args, session: aiohttp.ClientSession, worker_id: int, pdf_s3_path: str, pdf_local_path: str, page_num: int) -> PageResult:
    COMPLETION_URL = "http://localhost:30000/v1/chat/completions"
    MAX_RETRIES = 3
    
    attempt = 0
    await tracker.track_work(worker_id, f"{pdf_s3_path}-{page_num}", "started")

    while attempt < MAX_RETRIES:
        query = await build_page_query(
            pdf_local_path,
            page_num,
            args.target_longest_image_dim,
            args.target_anchor_text_len
        )

        try:
            async with session.post(COMPLETION_URL, json=query) as response:
                response.raise_for_status()

                base_response_data = await response.json()
                
                metrics.add_metrics(sglang_input_tokens=base_response_data["usage"].get("prompt_tokens", 0),
                                    sglang_output_tokens=base_response_data["usage"].get("completion_tokens", 0))

                model_response_json = json.loads(base_response_data["choices"][0]["message"]["content"])
                page_response = PageResponse(**model_response_json)

                await tracker.track_work(worker_id, f"{pdf_s3_path}-{page_num}", "finished")
                return PageResult(
                    pdf_s3_path,
                    page_num,
                    page_response,
                    input_tokens=base_response_data["usage"].get("prompt_tokens", 0),
                    output_tokens=base_response_data["usage"].get("completion_tokens", 0)
                )
        except aiohttp.ClientError as e:
            logger.warning(f"Client error on attempt {attempt} for {pdf_s3_path}-{page_num}: {e}")
            
            # Now we want to do exponential backoff, and not count this as an actual page retry
            # Page retrys are supposed to be for fixing bad results from the model, but actual requests to sglang 
            # are supposed to work. Probably this means that the server is just restarting
            logger.info(f"Sleeping for 5 seconds on {pdf_s3_path}-{page_num} to allow server restart")
            await asyncio.sleep(5)

        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error on attempt {attempt} for {pdf_s3_path}-{page_num}: {e}")
            attempt += 1
        except Exception as e:
            logger.warning(f"Unexpected error on attempt {attempt} for {pdf_s3_path}-{page_num}: {e}")
            attempt += 1

        if attempt >= MAX_RETRIES:
            logger.error(f"Failed to process {pdf_s3_path}-{page_num} after {MAX_RETRIES} attempts.")
            raise ValueError(f"Could not process {pdf_s3_path}-{page_num} after {MAX_RETRIES} attempts")

    await tracker.track_work(worker_id, f"{pdf_s3_path}-{page_num}", "errored")

async def process_pdf(args, worker_id: int, pdf_s3_path: str):
    with tempfile.NamedTemporaryFile("wb+", suffix=".pdf") as tf:
        # TODO Switch to aioboto3 or something
        data = await asyncio.to_thread(lambda: get_s3_bytes(pdf_s3, pdf_s3_path))
        tf.write(data)
        tf.flush()

        reader = PdfReader(tf.name)
        num_pages = reader.get_num_pages()

        # List to hold the tasks for processing each page
        page_tasks = []

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600), connector=aiohttp.TCPConnector(limit=10)) as session:
            for page_num in range(1, num_pages + 1):
                # Create a task for each page
                task = asyncio.create_task(process_page(args, session, worker_id, pdf_s3_path, tf.name, page_num))
                page_tasks.append(task)

            # Gather results from all page processing tasks
            try:
                page_results: list[PageResult] = await asyncio.gather(*page_tasks)
            except:
                logger.exception(f"Could not load page for {pdf_s3_path}, aborting document")
                return None


        # Build the document text and page spans
        document_text = ""
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
            pdf_page_spans.append([start_pos, current_char_pos, page_result.page_num])

        if not document_text:
            return None  # Return None if the document text is empty

        # Build the Dolma document
        metadata = {
            "Source-File": pdf_s3_path,
            "pdf-total-pages": num_pages,
            "total-input-tokens": sum(page.input_tokens for page in page_results),
            "total-output-tokens": sum(page.output_tokens for page in page_results)
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


async def worker(args, queue, semaphore, worker_id):
    while True:
        [work_hash, pdfs] = await queue.get()

        try:
            # Wait until allowed to proceed
            await semaphore.acquire()

            dolma_docs = await asyncio.gather(*[process_pdf(args, worker_id, pdf) for pdf in pdfs])
            dolma_docs = [doc for doc in dolma_docs if doc is not None]

            # Write the Dolma documents to a local temporary file in JSONL format
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tf:
                for doc in dolma_docs:
                    tf.write(json.dumps(doc))
                    tf.write('\n')
                tf.flush()

                # Define the output S3 path using the work_hash
                output_s3_path = os.path.join(args.workspace, 'dolma_documents', f'output_{work_hash}.jsonl')

                bucket, key = parse_s3_path(output_s3_path)
                workspace_s3.upload_file(tf.name, bucket, key)

            # Update finished token counts from successful documents
            metrics.add_metrics(finished_input_tokens=sum(doc["metadata"]["total-input-tokens"] for doc in dolma_docs),
                                finished_output_tokens=sum(doc["metadata"]["total-output-tokens"] for doc in dolma_docs))
  
            # Update last batch time
            last_batch_time = time.perf_counter()
        except Exception as e:
            logger.exception(f"Exception occurred while processing work_hash {work_hash}: {e}")
        finally:
            queue.task_done()


async def sglang_server_task(args, semaphore):
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
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        )

    # Make sure we kill this subprocess on exit
    def _kill_proc():
        proc.terminate()

    atexit.register(_kill_proc)

    last_running_req, last_queue_req = 0, 0 # To track transitions
    can_release_automatically = False
    last_semaphore_release = time.time()
    async def process_line(line):
        nonlocal last_running_req, last_queue_req, can_release_automatically, last_semaphore_release
        sglang_logger.info(line)

        match = re.search(r'#running-req: (\d+)', line)
        if match:
            last_running_req = int(match.group(1))

            if last_running_req > 0:
                can_release_automatically = True
        
        # Parse the line and update semaphore if necessary
        match = re.search(r'#queue-req: (\d+)', line)
        if match:
            queue_req = int(match.group(1))
            logger.info(f"sglang running req: {last_running_req} queue req: {queue_req}")
            
            if last_queue_req != 0 and queue_req == 0:
                # Release the semaphore when queue_req transitions from non-zero to zero
                if semaphore.locked():
                    semaphore.release()
                    last_semaphore_release = time.time()
                    logger.info("Semaphore released, allowing a worker to proceed.")

            last_queue_req = queue_req

        # And have a semaphore release automatically if there are no running requests for > 30 seconds
        if last_running_req == 0 and can_release_automatically and time.time() - last_semaphore_release > 30 and semaphore.locked():
            semaphore.release()
            last_semaphore_release = time.time()
            can_release_automatically = False
            logger.info("Semaphore released due to timeout, allowing a worker to proceed.")

    async def read_stream(stream):
        while True:
            line = await stream.readline()
            if not line:
                break
            line = line.decode('utf-8').rstrip()
            await process_line(line)

    # Start tasks to read stdout and stderr
    stdout_task = asyncio.create_task(read_stream(proc.stdout))
    stderr_task = asyncio.create_task(read_stream(proc.stderr))

    await proc.wait()
    await stdout_task
    await stderr_task


async def sglang_server_host(args, semaphore):
    while True:
        await sglang_server_task(args, semaphore)


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
            logger.warning(f"Attempt {attempt}: {e}")

        await asyncio.sleep(delay_sec)

    raise Exception("sglang server did not become ready after waiting.")


async def metrics_reporter():
    while True:
        # Leading newlines preserve table formatting in logs
        logger.info("\n" + str(metrics))
        logger.info("\n" + str(await tracker.get_status_table()))
        await asyncio.sleep(10)


async def main():
    parser = argparse.ArgumentParser(description='Manager for running millions of PDFs through a batch inference pipeline')
    parser.add_argument('workspace', help='The S3 path where work will be done e.g., s3://bucket/prefix/')
    parser.add_argument('--pdfs', help='Path to add pdfs stored in s3 to the workspace, can be a glob path s3://bucket/prefix/*.pdf or path to file containing list of pdf paths', default=None)
    parser.add_argument('--target_longest_image_dim', type=int, help='Dimension on longest side to use for rendering the pdf pages', default=1024)
    parser.add_argument('--target_anchor_text_len', type=int, help='Maximum amount of anchor text to use (characters)', default=6000)
    parser.add_argument('--workspace_profile', help='S3 configuration profile for accessing the workspace', default=None)
    parser.add_argument('--pdf_profile', help='S3 configuration profile for accessing the raw pdf documents', default=None)
    parser.add_argument('--group_size', type=int, default=20, help='Number of pdfs that will be part of each work item in the work queue.')
    parser.add_argument('--workers', type=int, default=3, help='Number of workers to run at a time')

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

    # Create a semaphore to control worker access
    # We only allow one worker to move forward with requests, until the server has no more requests in its queue
    # This lets us get full utilization by having many workers, but also to be outputting dolma docs as soon as possible
    # As soon as one worker is no longer saturating the gpu, the next one can start sending requests
    semaphore = asyncio.Semaphore(1)

    sglang_server = asyncio.create_task(sglang_server_host(args, semaphore))

    work_queue = await load_pdf_work_queue(args)
    logger.info(f"Work queue prepared with {work_queue.qsize()} items")

    await sglang_server_ready()

    metrics_task = asyncio.create_task(metrics_reporter())

    # Create worker tasks to process the queue concurrently.
    worker_tasks = []
    for i in range(args.workers):
        task = asyncio.create_task(worker(args, work_queue, semaphore, worker_id=i))
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

    metrics_task.cancel()
    await metrics_task
    

if __name__ == "__main__":
    asyncio.run(main())

    # TODO
    # If there is a beaker flag, then your job is to trigger this script with N replicas on beaker
    # If not, then your job is to do the actual work

    # TODO
    # Possible future addon, in beaker, discover other nodes on this same job
    # Send them a message when you take a work item off the queue
