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
import random
import re

from tqdm import tqdm
from io import BytesIO
from PIL import Image
from pypdf import PdfReader
from functools import partial
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ProcessPoolExecutor

from pdelfin.s3_utils import expand_s3_glob, get_s3_bytes, get_s3_bytes_with_backoff, parse_s3_path, download_zstd_csv, upload_zstd_csv, download_directory
from pdelfin.data.renderpdf import render_pdf_to_base64png
from pdelfin.prompts import build_finetuning_prompt, PageResponse
from pdelfin.prompts.anchor import get_anchor_text
from pdelfin.check import check_poppler_version
from pdelfin.metrics import MetricsKeeper, WorkerTracker
from pdelfin.version import VERSION

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

SGLANG_SERVER_PORT = 30002

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

    sample_size = min(100, len(new_pdfs))
    sampled_pdfs = random.sample(list(new_pdfs), sample_size)

    page_counts = []

    for pdf in tqdm(sampled_pdfs, desc="Sampling PDFs to calculate optimial length"):
        try:
            # Download the PDF to a temp file
            with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_file:
                s3_bucket, s3_key = parse_s3_path(pdf)
                pdf_s3.download_fileobj(s3_bucket, s3_key, tmp_file)
                tmp_file.flush()
                reader = PdfReader(tmp_file.name)
                page_counts.append(len(reader.pages))
        except Exception as e:
            logger.warning(f"Failed to read {pdf}: {e}")

    if page_counts:
        avg_pages_per_pdf = sum(page_counts) / len(page_counts)
    else:
        logger.warning("Could not read any PDFs to estimate average page count.")
        avg_pages_per_pdf = 10  # Default to 10 pages per PDF if sampling fails

    group_size = max(1, int(args.pages_per_group / avg_pages_per_pdf))
    logger.info(f"Calculated group_size: {group_size} based on average pages per PDF: {avg_pages_per_pdf:.2f}")

    new_groups = []
    current_group = []
    for pdf in sorted(new_pdfs):  # Sort for consistency
        current_group.append(pdf)
        if len(current_group) == group_size:
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
    output_glob = os.path.join(args.workspace, "dolma_documents", "*.jsonl")

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
    #remaining_work_hashes = set(["0e779f21fbb75d38ed4242c7e5fe57fa9a636bac"])
    remaining_work_queue = {
        hash_: work_queue[hash_]
        for hash_ in remaining_work_hashes
    }

    # Populate the asyncio.Queue with remaining work
    queue = asyncio.Queue()
    shuffled_items = list(remaining_work_queue.items())
    random.shuffle(shuffled_items)

    for work, pdfs in shuffled_items:
        await queue.put((work, pdfs))

    return queue

async def work_item_completed(args, work_hash: str) -> bool:
    # Check if work item has already been completed
    output_s3_path = os.path.join(args.workspace, 'dolma_documents', f'output_{work_hash}.jsonl')
    bucket, key = parse_s3_path(output_s3_path)
    
    try:
        # Check if the output file already exists
        await asyncio.to_thread(workspace_s3.head_object, Bucket=bucket, Key=key)
        return True
    except workspace_s3.exceptions.ClientError as e:
        pass

    return False


async def process_page(args, session: aiohttp.ClientSession, worker_id: int, pdf_s3_path: str, pdf_local_path: str, page_num: int) -> PageResult:
    COMPLETION_URL = f"http://localhost:{SGLANG_SERVER_PORT}/v1/chat/completions"
    MAX_RETRIES = 3
    
    exponential_backoffs = 0
    local_anchor_text_len = args.target_anchor_text_len
    attempt = 0
    await tracker.track_work(worker_id, f"{pdf_s3_path}-{page_num}", "started")

    while attempt < MAX_RETRIES:
        query = await build_page_query(
            pdf_local_path,
            page_num,
            args.target_longest_image_dim,
            local_anchor_text_len
        )

        try:
            async with session.post(COMPLETION_URL, json=query) as response:
                response.raise_for_status()

                base_response_data = await response.json()

                if base_response_data["usage"]["total_tokens"] > args.model_max_context:
                    local_anchor_text_len = max(1, local_anchor_text_len // 2)
                    logger.info(f"Reducing anchor text len to {local_anchor_text_len} for {pdf_s3_path}-{page_num}")
                    raise ValueError(f"Response exceeded model_max_context, cannot use this response")
                
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
            sleep_delay = 10 * (2 ** exponential_backoffs)
            exponential_backoffs += 1
            logger.info(f"Sleeping for {sleep_delay} seconds on {pdf_s3_path}-{page_num} to allow server restart")
            await asyncio.sleep(sleep_delay)
        except asyncio.CancelledError:
            logger.info(f"Process page {pdf_s3_path}-{page_num} cancelled")
            await tracker.track_work(worker_id, f"{pdf_s3_path}-{page_num}", "cancelled")
            raise
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error on attempt {attempt} for {pdf_s3_path}-{page_num}: {e}")
            attempt += 1
        except ValueError as e:
            logger.warning(f"ValueError on attempt {attempt} for {pdf_s3_path}-{page_num}: {type(e)} - {e}")
            attempt += 1
        except Exception as e:
            logger.warning(f"Unexpected error on attempt {attempt} for {pdf_s3_path}-{page_num}: {type(e)} - {e}")
            attempt += 1

    logger.error(f"Failed to process {pdf_s3_path}-{page_num} after {MAX_RETRIES} attempts.")
    await tracker.track_work(worker_id, f"{pdf_s3_path}-{page_num}", "errored")
    raise ValueError(f"Could not process {pdf_s3_path}-{page_num} after {MAX_RETRIES} attempts")


async def process_pdf(args, session: aiohttp.ClientSession, worker_id: int, pdf_s3_path: str):
    with tempfile.NamedTemporaryFile("wb+", suffix=".pdf") as tf:
        # TODO Switch to aioboto3 or something
        data = await asyncio.to_thread(lambda: get_s3_bytes_with_backoff(pdf_s3, pdf_s3_path))
        tf.write(data)
        tf.flush()

        try:
            reader = PdfReader(tf.name)
            num_pages = reader.get_num_pages()
        except:
            logger.exception(f"Could not count number of pages for {pdf_s3_path}, aborting document")
            return None

        logger.info(f"Got {num_pages} pages to do for {pdf_s3_path} in worker {worker_id}")

        # List to hold the tasks for processing each page
        page_tasks = []
        page_results = []

        try:
            async with asyncio.TaskGroup() as tg:
                for page_num in range(1, num_pages + 1):
                    task = tg.create_task(process_page(args, session, worker_id, pdf_s3_path, tf.name, page_num))
                    page_tasks.append(task)

            # Collect the results from the entire task group, assuming no exceptions
            page_results = [task.result() for task in page_tasks]
            return build_dolma_document(pdf_s3_path, page_results)
        except Exception as e:
            logger.exception(f"Exception in process_pdf for {pdf_s3_path}: {e}")
            # You can't build a dolma doc with even 1 failed page, so just get out of here
            # However, you don't want to propagate an exception higher up and cancel the entire work_group
            return None


def build_dolma_document(pdf_s3_path, page_results):
    # Build the document text and page spans
    document_text = ""
    pdf_page_spans = []
    current_char_pos = 0

    for index, page_result in enumerate(page_results):
        if page_result.response.natural_text is not None:
            content = page_result.response.natural_text + ("\n" if index < len(page_results) - 1 else "")
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
        "pdf-total-pages": len(page_results),
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
            await tracker.clear_work(worker_id)

            # Wait until allowed to proceed
            await semaphore.acquire()

            if await work_item_completed(args, work_hash):
                logger.info(f"Work {work_hash} was already completed, skipping")
                continue
            else:
                logger.info(f"Proceeding with {work_hash} on worker {worker_id}")

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600), 
                                             connector=aiohttp.TCPConnector(limit=1000)) as session:
                async with asyncio.TaskGroup() as tg:      
                    dolma_tasks = [tg.create_task(process_pdf(args, session, worker_id, pdf)) for pdf in pdfs]

            dolma_docs = []
            for task in dolma_tasks:
                try:
                    result = task.result()
                except:
                    # some dolma doc creations may have failed
                    pass

                if result is not None:
                    dolma_docs.append(result)

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
    download_directory(args.model, model_cache_dir)

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

        # TODO Had to comment this out, I thought it would be good to enforce a context limit on the server side, but it causes crashes
        #"--context-length", str(args.model_max_context),

        "--port", str(SGLANG_SERVER_PORT),
        "--log-level-http", "warning",
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

        # And have a semaphore release automatically if there are no queued requests for > 30 seconds
        if last_queue_req == 0 and can_release_automatically and time.time() - last_semaphore_release > 30 and semaphore.locked():
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
        logger.warning("SGLang server task ended")


async def sglang_server_ready():
    max_attempts = 300
    delay_sec = 1
    url = f'http://localhost:{SGLANG_SERVER_PORT}/v1/models'

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


async def metrics_reporter(queue):
    while True:
        # Leading newlines preserve table formatting in logs
        logger.info(f"Queue remaining: {queue.qsize()}")
        logger.info("\n" + str(metrics))
        logger.info("\n" + str(await tracker.get_status_table()))
        await asyncio.sleep(10)



def submit_beaker_job(args):
    from beaker import (
        Beaker,
        Constraints,
        DataMount,
        DataSource,
        EnvVar,
        ExperimentSpec,
        ImageSource,
        Priority,
        ResultSpec,
        SecretNotFound,
        TaskContext,
        TaskResources,
        TaskSpec,
    )
    
    b = Beaker.from_env(default_workspace=args.beaker_workspace)
    account = b.account.whoami()
    owner = account.name
    beaker_image = f"jakep/pdelfin-inference-{VERSION}"

    task_name = f"pdelfin-{os.path.basename(args.workspace.rstrip('/'))}"

    args_list = [arg for arg in sys.argv[1:] if arg != "--beaker"]

    try:
        b.secret.get(f"{owner}-WEKA_ACCESS_KEY_ID", args.beaker_workspace)
        b.secret.get(f"{owner}-WEKA_SECRET_ACCESS_KEY", args.beaker_workspace)
        b.secret.get(f"{owner}-AWS_CREDENTIALS_FILE", args.beaker_workspace)
    except SecretNotFound:
        print(f"Expected beaker secrets for accessing Weka and S3 are not found. Are you okay to write those to your beaker workspace {args.beaker_workspace}? [y/n]")

        if input().strip().lower() != "y":
            print("Exiting...")
            sys.exit(1)

        b.secret.write(f"{owner}-WEKA_ACCESS_KEY_ID", os.environ.get("WEKA_ACCESS_KEY_ID", ""), args.beaker_workspace)
        b.secret.write(f"{owner}-WEKA_SECRET_ACCESS_KEY", os.environ.get("WEKA_SECRET_ACCESS_KEY", ""), args.beaker_workspace)
        b.secret.write(f"{owner}-AWS_CREDENTIALS_FILE", open(os.path.join(os.path.expanduser('~'), '.aws', 'credentials')).read(), args.beaker_workspace)


    # Create the experiment spec
    experiment_spec = ExperimentSpec(
        budget="ai2/oe-data",
        description=task_name,
        tasks=[
            TaskSpec(
                name=task_name,
                propagate_failure=False,
                propagate_preemption=False,
                replicas=args.beaker_gpus,
                context=TaskContext(
                    priority=Priority(args.beaker_priority),
                    preemptible=True,
                ),
                image=ImageSource(beaker=beaker_image),
                command=["python", "-m", "pdelfin.beakerpipeline"] + args_list,
                env_vars=[
                    EnvVar(name="BEAKER_JOB_NAME", value=task_name),
                    EnvVar(name="OWNER", value=owner),
                    EnvVar(name="WEKA_ACCESS_KEY_ID", secret=f"{owner}-WEKA_ACCESS_KEY_ID"),
                    EnvVar(name="WEKA_SECRET_ACCESS_KEY", secret=f"{owner}-WEKA_SECRET_ACCESS_KEY"),
                    EnvVar(name="AWS_CREDENTIALS_FILE", secret=f"{owner}-AWS_CREDENTIALS_FILE"),
                ],
                resources=TaskResources(gpu_count=1),
                constraints=Constraints(cluster=args.beaker_cluster),
                result=ResultSpec(path="/noop-results"),
            )
        ],
    )
    
    experiment_data = b.experiment.create(spec=experiment_spec, workspace=args.beaker_workspace)
    
    print(f"Experiment URL: https://beaker.org/ex/{experiment_data.id}")


async def main():
    parser = argparse.ArgumentParser(description='Manager for running millions of PDFs through a batch inference pipeline')
    parser.add_argument('workspace', help='The S3 path where work will be done e.g., s3://bucket/prefix/')
    parser.add_argument('--pdfs', help='Path to add pdfs stored in s3 to the workspace, can be a glob path s3://bucket/prefix/*.pdf or path to file containing list of pdf paths', default=None)
    parser.add_argument('--workspace_profile', help='S3 configuration profile for accessing the workspace', default=None)
    parser.add_argument('--pdf_profile', help='S3 configuration profile for accessing the raw pdf documents', default=None)
    parser.add_argument('--pages_per_group', type=int, default=500, help='Aiming for this many pdf pages per work item group')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers to run at a time')

    # Model parameters
    parser.add_argument('--model', help='List of paths where you can find the model to convert this pdf. You can specify several different paths here, and the script will try to use the one which is fastest to access',
                         default=["weka://oe-data-default/jakep/Qwen_Qwen2-VL-7B-Instruct-e4ecf8-01JAH8GMWHTJ376S2N7ETXRXH4/best_bf16/",
                                  "gs://ai2-oe-data/jakep/experiments/qwen2vl-pdf/v1/models/jakep/Qwen_Qwen2-VL-7B-Instruct-e4ecf8-01JAH8GMWHTJ376S2N7ETXRXH4/checkpoint-9500/bf16/",
                                  "s3://ai2-oe-data/jakep/experiments/qwen2vl-pdf/v1/models/jakep/Qwen_Qwen2-VL-7B-Instruct-e4ecf8-01JAH8GMWHTJ376S2N7ETXRXH4/checkpoint-9500/bf16/"])
    parser.add_argument('--model_max_context', type=int, default="8192", help="Maximum context length that the model was fine tuned under")
    parser.add_argument('--model_chat_template', type=str, default="qwen2-vl", help="Chat template to pass to sglang server")
    parser.add_argument('--target_longest_image_dim', type=int, help='Dimension on longest side to use for rendering the pdf pages', default=1024)
    parser.add_argument('--target_anchor_text_len', type=int, help='Maximum amount of anchor text to use (characters)', default=6000)

    # Beaker/job running stuff
    parser.add_argument('--beaker', action='store_true', help='Submit this job to beaker instead of running locally')
    parser.add_argument('--beaker_workspace', help='Beaker workspace to submit to', default='ai2/pdelfin')
    parser.add_argument('--beaker_cluster', help='Beaker clusters you want to run on', default=["ai2/jupiter-cirrascale-2", "ai2/pluto-cirrascale", "ai2/saturn-cirrascale"])
    parser.add_argument('--beaker_gpus', type=int, default=1, help="Number of gpu replicas to run")
    parser.add_argument('--beaker_priority', type=str, default="normal", help="Beaker priority level for the job")
    args = parser.parse_args()

    global workspace_s3, pdf_s3

    # setup the job to work in beaker environment, load secrets, adjust logging, etc.
    if "BEAKER_JOB_NAME" in os.environ:
        sglang_logger.addHandler(console_handler)
        cred_path = os.path.join(os.path.expanduser('~'), '.aws', 'credentials')
        os.makedirs(os.path.dirname(cred_path), exist_ok=True)
        with open(cred_path, "w") as f:
            f.write(os.environ.get("AWS_CREDENTIALS_FILE"))
        workspace_s3 = boto3.client('s3')
        pdf_s3 = boto3.client('s3')

    if args.workspace_profile:
        workspace_session = boto3.Session(profile_name=args.workspace_profile)
        workspace_s3 = workspace_session.client("s3")

    if args.pdf_profile:
        pdf_session = boto3.Session(profile_name=args.pdf_profile)
        pdf_s3 = pdf_session.client("s3")

    check_poppler_version()

    if args.pdfs:
        logger.info("Got --pdfs argument, going to add to the work queue")
        await populate_pdf_work_queue(args)

    if args.beaker:
        submit_beaker_job(args)
        return

    logger.info(f"Starting pipeline with PID {os.getpid()}")

    # Create a semaphore to control worker access
    # We only allow one worker to move forward with requests, until the server has no more requests in its queue
    # This lets us get full utilization by having many workers, but also to be outputting dolma docs as soon as possible
    # As soon as one worker is no longer saturating the gpu, the next one can start sending requests
    semaphore = asyncio.Semaphore(1)

    sglang_server = asyncio.create_task(sglang_server_host(args, semaphore))

    work_queue = await load_pdf_work_queue(args)
    logger.info(f"Work queue prepared with {work_queue.qsize()} items")

    await sglang_server_ready()

    metrics_task = asyncio.create_task(metrics_reporter(work_queue))

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
    # Possible future addon, in beaker, discover other nodes on this same job
    # Send them a message when you take a work item off the queue
