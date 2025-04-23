#!/usr/bin/env python3
"""
Tagging pipeline for Dolma JSONL datasets.

For each .jsonl, .jsonl.gz, or .jsonl.ztd file under the dataset/documents folder,
this script issues a simple SGLang completion per record (e.g., "Is this document in English?"),
collects the yes/no answers, and writes corresponding Dolma attributes JSONL files under
scratch/attributes/, mirroring the input structure.
"""
import argparse
import asyncio
import atexit
import base64
import datetime
import hashlib
import json
import logging
import multiprocessing
import os
import random
import re
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from functools import cache, partial
from io import BytesIO
from urllib.parse import urlparse

import boto3
import httpx
import torch
from botocore.exceptions import ClientError
from huggingface_hub import snapshot_download
from PIL import Image
from pypdf import PdfReader
from tqdm import tqdm

from olmocr.check import (
    check_poppler_version,
    check_sglang_version,
    check_torch_gpu_available,
)
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.filter.filter import Language, PdfFilter
from olmocr.image_utils import convert_image_to_pdf_bytes, is_jpeg, is_png
from olmocr.metrics import MetricsKeeper, WorkerTracker
from olmocr.prompts import PageResponse, build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text
from olmocr.s3_utils import (
    download_directory,
    download_zstd_csv,
    expand_s3_glob,
    get_s3_bytes,
    get_s3_bytes_with_backoff,
    parse_s3_path,
)
from olmocr.version import VERSION
from olmocr.work_queue import LocalWorkQueue, S3WorkQueue, WorkQueue


# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

sglang_logger = logging.getLogger("sglang")
sglang_logger.propagate = False

file_handler = logging.FileHandler("olmocr-pipeline-debug.log", mode="a")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
sglang_logger.addHandler(file_handler)


# Default port; overridden by --port
SGLANG_SERVER_PORT = 30024

# Global variables for token statistics
metrics = MetricsKeeper(window=60 * 5)
tracker = WorkerTracker()


# Process pool for offloading cpu bound work, like calculating anchor texts, max 32 workers, otherwise it can spawn way too many workers on a big machine
process_pool = ProcessPoolExecutor(max_workers=min(multiprocessing.cpu_count() // 2 + 1, 32), mp_context=multiprocessing.get_context("spawn"))


async def process_file(args, worker_id: int, file_uri: str):
    """
    Download a JSONL file, query SGLang per record, and write attributes.
    """
    # Fetch raw bytes (S3 or local)
    if file_uri.startswith("s3://"):
        raw = await asyncio.to_thread(get_s3_bytes_with_backoff, dataset_s3, file_uri)
    else:
        raise NotImplementedError()
    
    return

async def worker(args, work_queue: WorkQueue, semaphore, worker_id):
    while True:
        # Wait until allowed to proceed
        await semaphore.acquire()

        work_item = await work_queue.get_work()

        if work_item is None:
            logger.info(f"Worker {worker_id} exiting due to empty queue")
            semaphore.release()
            break

        assert len(work_item.work_paths) == 1, "We are assuming 1 work path per work item in this pipeline"

        logger.info(f"Worker {worker_id} processing work item {work_item.work_paths[0]}")
        await tracker.clear_work(worker_id)

        try:
            json_attributes = await process_file(args, worker_id, work_item.work_paths[0])

            # Write the attributes to the results folder to indiciate the work is done
            

            await work_queue.mark_done(work_item)
        except Exception as e:
            logger.exception(f"Exception occurred while processing work_hash {work_item.hash}: {e}")
        finally:
            semaphore.release()


async def sglang_server_task(model_name_or_path, args, semaphore):
    # Check GPU memory, lower mem devices need a bit less KV cache space because the VLM takes additional memory
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
    mem_fraction_arg = ["--mem-fraction-static", "0.80"] if gpu_memory < 60 else []

    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_name_or_path,
        "--chat-template",
        args.model_chat_template,
        # "--context-length", str(args.model_max_context),  # Commented out due to crashes
        "--port",
        str(SGLANG_SERVER_PORT),
        "--log-level-http",
        "warning",
    ]
    cmd.extend(mem_fraction_arg)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Ensure the subprocess is terminated on exit
    def _kill_proc():
        proc.terminate()

    atexit.register(_kill_proc)

    # Shared variables between tasks
    last_running_req, last_queue_req = 0, 0
    server_printed_ready_message = False
    last_semaphore_release = time.time()

    async def process_line(line):
        nonlocal last_running_req, last_queue_req, last_semaphore_release, server_printed_ready_message
        sglang_logger.info(line)

        # if the server hasn't initialized yet, log all the lines to the main logger also, so that the user
        # can see any warnings/errors more easily
        if not server_printed_ready_message:
            logger.info(line)

        if "Detected errors during sampling" in line:
            logger.error("Cannot continue, sampling errors detected, model is probably corrupt")
            sys.exit(1)

        # TODO, need to trace down this issue in sglang itself, but it will otherwise cause the server to lock up
        if "IndexError: list index out of range" in line:
            logger.error("IndexError in model, restarting server")
            proc.terminate()

        if not server_printed_ready_message and "The server is fired up and ready to roll!" in line:
            server_printed_ready_message = True
            last_semaphore_release = time.time()

        match = re.search(r"#running-req: (\d+)", line)
        if match:
            last_running_req = int(match.group(1))

        match = re.search(r"#queue-req: (\d+)", line)
        if match:
            last_queue_req = int(match.group(1))
            logger.info(f"sglang running req: {last_running_req} queue req: {last_queue_req}")

    async def read_stream(stream):
        while True:
            line = await stream.readline()
            if not line:
                break
            try:
                line = line.decode("utf-8").rstrip()
                await process_line(line)
            except Exception as ex:
                logger.warning(f"Got {ex} when reading log line from inference server, skipping")

    async def timeout_task():
        nonlocal last_running_req, last_queue_req, last_semaphore_release
        try:
            while True:
                await asyncio.sleep(1)
                if server_printed_ready_message and last_queue_req == 0 and time.time() - last_semaphore_release > 30 and semaphore.locked():
                    semaphore.release()
                    last_semaphore_release = time.time()
                    logger.info("Semaphore released, allowing a worker to proceed.")
        except asyncio.CancelledError:
            pass  # Clean up if the task is cancelled

    # Start tasks to read stdout, stderr, and handle timeout logic
    stdout_task = asyncio.create_task(read_stream(proc.stdout))
    stderr_task = asyncio.create_task(read_stream(proc.stderr))
    timeout_task = asyncio.create_task(timeout_task())

    try:
        await proc.wait()
    except asyncio.CancelledError:
        logger.info("Got cancellation request for SGLang server")
        proc.terminate()
        raise

    timeout_task.cancel()
    await asyncio.gather(stdout_task, stderr_task, timeout_task, return_exceptions=True)


async def sglang_server_host(model_name_or_path, args, semaphore):
    MAX_RETRIES = 5
    retry = 0

    while retry < MAX_RETRIES:
        await sglang_server_task(model_name_or_path, args, semaphore)
        logger.warning("SGLang server task ended")
        retry += 1

    if retry >= MAX_RETRIES:
        logger.error(f"Ended up starting the sglang server more than {retry} times, cancelling pipeline")
        logger.error("")
        logger.error("Please make sure sglang is installed according to the latest instructions here: https://docs.sglang.ai/start/install.html")
        sys.exit(1)


async def sglang_server_ready():
    max_attempts = 300
    delay_sec = 1
    url = f"http://localhost:{SGLANG_SERVER_PORT}/v1/models"

    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient() as session:
                response = await session.get(url)

                if response.status_code == 200:
                    logger.info("sglang server is ready.")
                    return
                else:
                    logger.info(f"Attempt {attempt}: Unexpected status code {response.status_code}")
        except Exception:
            logger.warning(f"Attempt {attempt}: Please wait for sglang server to become ready...")

        await asyncio.sleep(delay_sec)

    raise Exception("sglang server did not become ready after waiting.")


async def download_model(model_name_or_path: str):
    if model_name_or_path.startswith("s3://") or model_name_or_path.startswith("gs://") or model_name_or_path.startswith("weka://"):
        logger.info(f"Downloading model directory from '{model_name_or_path}'")
        model_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "olmocr", "model")
        download_directory([model_name_or_path], model_cache_dir)
        return model_cache_dir
    elif os.path.isabs(model_name_or_path) and os.path.isdir(model_name_or_path):
        logger.info(f"Using local model path at '{model_name_or_path}'")
        return model_name_or_path
    else:
        logger.info(f"Downloading model with hugging face '{model_name_or_path}'")
        snapshot_download(repo_id=model_name_or_path)
        return model_name_or_path


async def metrics_reporter(work_queue):
    while True:
        # Leading newlines preserve table formatting in logs
        logger.info(f"Queue remaining: {work_queue.size}")
        logger.info("\n" + str(metrics))
        logger.info("\n" + str(await tracker.get_status_table()))
        await asyncio.sleep(10)


def submit_beaker_job(args):
    from beaker import (  # type: ignore
        Beaker,
        Constraints,
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
    beaker_image = f"jakep/olmocr-inference-{VERSION}"

    task_name = f"olmocr-{os.path.basename(args.dataset.rstrip('/'))}"

    # Take out --beaker flag so the workers will just run things
    args_list = [arg for arg in sys.argv[1:] if arg != "--beaker"]

    # Take out the --pdfs [arg] or --pdfs=[arg], since the queue is populated locally
    args_list = [arg for i, arg in enumerate(args_list) if not (arg.startswith("--pdfs") or (i > 0 and args_list[i - 1] == "--pdfs"))]

    try:
        b.secret.get(f"{owner}-WEKA_ACCESS_KEY_ID", args.beaker_workspace)
        b.secret.get(f"{owner}-WEKA_SECRET_ACCESS_KEY", args.beaker_workspace)
        b.secret.get(f"{owner}-AWS_CREDENTIALS_FILE", args.beaker_workspace)
    except SecretNotFound:
        print(
            f"Expected beaker secrets for accessing Weka and S3 are not found. Are you okay to write those to your beaker workspace {args.beaker_workspace}? [y/n]"
        )

        if input().strip().lower() != "y":
            print("Exiting...")
            sys.exit(1)

        b.secret.write(f"{owner}-WEKA_ACCESS_KEY_ID", os.environ.get("WEKA_ACCESS_KEY_ID", ""), args.beaker_workspace)
        b.secret.write(f"{owner}-WEKA_SECRET_ACCESS_KEY", os.environ.get("WEKA_SECRET_ACCESS_KEY", ""), args.beaker_workspace)
        b.secret.write(
            f"{owner}-AWS_CREDENTIALS_FILE",
            open(os.path.join(os.path.expanduser("~"), ".aws", "credentials")).read(),
            args.beaker_workspace,
        )

    env_var_secrets = [
        EnvVar(name="WEKA_ACCESS_KEY_ID", secret=f"{owner}-WEKA_ACCESS_KEY_ID"),
        EnvVar(name="WEKA_SECRET_ACCESS_KEY", secret=f"{owner}-WEKA_SECRET_ACCESS_KEY"),
        EnvVar(name="AWS_CREDENTIALS_FILE", secret=f"{owner}-AWS_CREDENTIALS_FILE"),
    ]

    try:
        b.secret.get("OLMOCR_PREVIEW_HF_TOKEN", args.beaker_workspace)
        env_var_secrets.append(EnvVar(name="HF_TOKEN", secret="OLMOCR_PREVIEW_HF_TOKEN"))
    except SecretNotFound:
        pass

    try:
        b.secret.get("OE_DATA_GCS_SA_KEY", args.beaker_workspace)
        env_var_secrets.append(EnvVar(name="GOOGLE_APPLICATION_CREDENTIALS_FILE", secret="OE_DATA_GCS_SA_KEY"))
    except SecretNotFound:
        print("Input the olmo-gcs SA key if you would like to load weights from gcs (end with a double newline):")
        lines = []
        prev_empty = False
        for line in iter(input, None):
            if not line and prev_empty:
                break
            prev_empty = not line
            lines.append(line)
        gcs_sa_key = "\n".join(lines[:-1]).strip()  # Remove the last empty line
        if gcs_sa_key:
            b.secret.write("OE_DATA_GCS_SA_KEY", gcs_sa_key, args.beaker_workspace)
            env_var_secrets.append(EnvVar(name="GOOGLE_APPLICATION_CREDENTIALS_FILE", secret="OE_DATA_GCS_SA_KEY"))

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
                command=["python", "-m", "scripts/tagging_pipeline.py"] + args_list,
                env_vars=[EnvVar(name="BEAKER_JOB_NAME", value=task_name), EnvVar(name="OWNER", value=owner)] + env_var_secrets,
                resources=TaskResources(gpu_count=1),
                constraints=Constraints(cluster=args.beaker_cluster if isinstance(args.beaker_cluster, list) else [args.beaker_cluster]),
                result=ResultSpec(path="/noop-results"),
            )
        ],
    )

    experiment_data = b.experiment.create(spec=experiment_spec, workspace=args.beaker_workspace)

    print(f"Experiment URL: https://beaker.org/ex/{experiment_data.id}")


async def main():
    parser = argparse.ArgumentParser(description="Tagging pipeline for Dolma JSONL dataset")
    parser.add_argument("dataset", help="Dolma dataset root (local or s3://) with documents/ folder")
    parser.add_argument("scratch", help="Scratch workspace (local dir or s3://)")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent workers")
    parser.add_argument("--model", default="google/gemma-3-4b-it", help="SGLang model path or name")

    # Beaker/job running stuff
    parser.add_argument("--beaker", action="store_true", help="Submit this job to beaker instead of running locally")
    parser.add_argument("--beaker_workspace", help="Beaker workspace to submit to", default="ai2/olmocr")
    parser.add_argument(
        "--beaker_cluster",
        help="Beaker clusters you want to run on",
        default=["ai2/jupiter-cirrascale-2", "ai2/ceres-cirrascale", "ai2/neptune-cirrascale", "ai2/saturn-cirrascale", "ai2/augusta-google-1"],
    )
    parser.add_argument("--beaker_gpus", type=int, default=1, help="Number of gpu replicas to run")
    parser.add_argument("--beaker_priority", type=str, default="normal", help="Beaker priority level for the job")

    parser.add_argument("--port", type=int, default=30024, help="Port for SGLang server")
    args = parser.parse_args()

    global SGLANG_SERVER_PORT, workspace_s3, dataset_s3
    SGLANG_SERVER_PORT = args.port
    workspace_s3 = boto3.client("s3")
    dataset_s3 = boto3.client("s3")

    # setup the job to work in beaker environment, load secrets, adjust logging, etc.
    if "BEAKER_JOB_NAME" in os.environ:
        sglang_logger.addHandler(console_handler)
        cred_path = os.path.join(os.path.expanduser("~"), ".aws", "credentials")
        os.makedirs(os.path.dirname(cred_path), exist_ok=True)
        with open(cred_path, "w") as f:
            f.write(os.environ.get("AWS_CREDENTIALS_FILE"))
        cred_path = os.path.join(os.path.expanduser("~"), ".gcs", "credentials")
        os.makedirs(os.path.dirname(cred_path), exist_ok=True)
        with open(cred_path, "w") as f:
            f.write(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_FILE"))
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
        workspace_s3 = boto3.client("s3")
        pdf_s3 = boto3.client("s3")

        # Wait a little bit so that not all beaker jobs in a task start at the same time and download the model at the same time
        replica_count = int(os.environ.get("BEAKER_REPLICA_COUNT", "1"))
        interval = 10 if (replica_count - 1) * 10 <= 240 else 240 / max(1, replica_count - 1)
        sleep_time = int(int(os.environ.get("BEAKER_REPLICA_RANK", "0")) * interval)
        logger.info(f"Beaker job sleeping for {sleep_time} seconds to stagger model downloads")
        await asyncio.sleep(sleep_time)

    # Initialize work queue
    if args.scratch.startswith("s3://"):
        work_queue = S3WorkQueue(workspace_s3, args.scratch)
    else:
        work_queue = LocalWorkQueue(args.scratch)

    # Discover input files
    files = set()
    if args.dataset.startswith("s3://"):
        pattern = args.dataset.rstrip("/") + "/documents/*.jsonl*"
        matched = expand_s3_glob(dataset_s3, pattern)
        files = set(matched.keys())
    else:
        docs_dir = os.path.join(args.dataset, "documents")
        for root, _, fns in os.walk(docs_dir):
            for fn in fns:
                if fn.endswith((".jsonl", ".jsonl.gz", ".jsonl.ztd")):
                    files.add(os.path.join(root, fn))

    # Populate the work queue if needed
    await work_queue.populate_queue(list(files), items_per_group=1)

    if args.beaker:
        submit_beaker_job(args)
        return

    # If you get this far, then you are doing inference and need a GPU
    # check_sglang_version()
    # check_torch_gpu_available()

    logger.info(f"Starting pipeline with PID {os.getpid()}")

    # Download the model before you do anything else
    model_name_or_path = await download_model(args.model)

    # Initialize the work queue
    qsize = await work_queue.initialize_queue()

    if qsize == 0:
        logger.info("No work to do, exiting")
        return

    # Create a semaphore to control worker access
    # We only allow one worker to move forward with requests, until the server has no more requests in its queue
    # This lets us get full utilization by having many workers, but also to be outputting dolma docs as soon as possible
    # As soon as one worker is no longer saturating the gpu, the next one can start sending requests
    semaphore = asyncio.Semaphore(1)

    # sglang_server = asyncio.create_task(sglang_server_host(model_name_or_path, args, semaphore))

    # await sglang_server_ready()

    metrics_task = asyncio.create_task(metrics_reporter(work_queue))

    # Create worker tasks to process the queue concurrently.
    worker_tasks = []
    for i in range(args.workers):
        task = asyncio.create_task(worker(args, work_queue, semaphore, worker_id=i))
        worker_tasks.append(task)

    # Wait for all worker tasks to finish
    await asyncio.gather(*worker_tasks)

    # Wait for server to stop
    process_pool.shutdown(wait=False)

    # sglang_server.cancel()
    metrics_task.cancel()
    logger.info("Work done")


if __name__ == "__main__":
    asyncio.run(main())