import os
import random
import logging
import hashlib
import tempfile
from typing import Optional, Tuple, List, Dict, Set
from dataclasses import dataclass
import asyncio
from functools import partial

from pdelfin.s3_utils import (
    expand_s3_glob,
    download_zstd_csv,
    upload_zstd_csv,
    parse_s3_path
)
from pypdf import PdfReader

logger = logging.getLogger(__name__)

@dataclass
class WorkItem:
    """Represents a single work item in the queue"""
    hash: str
    s3_work_paths: List[str]

class S3WorkQueue:
    """
    Manages a work queue stored in S3 that coordinates work across multiple workers.
    The queue maintains a list of work items, where each work item is a group of s3 paths
    that should be processed together.

    Each work item gets a hash, and completed work items will have their results 
    stored in s3://workspace_path/results/output_[hash].jsonl

    This is the ground source of truth about which work items are done.

    When a worker takes an item off the queue, it will write an empty s3 file to
    s3://workspace_path/worker_locks/output_[hash].jsonl

    The queue gets randomized on each worker, so workers pull random work items to operate on.
    As you pull an item, we will check to see if it has been completed. If yes,
    then it will immediately fetch the next item. If a lock file was created within a configurable
    timeout (30 mins by default), then that work item is also skipped.

    The lock will will be deleted once the worker is done with that item.
    """
    def __init__(self, s3_client, workspace_path: str):
        """
        Initialize the work queue.
        
        Args:
            s3_client: Boto3 S3 client to use for operations
            workspace_path: S3 path where work queue and results are stored
        """
        self.s3_client = s3_client
        self.workspace_path = workspace_path.rstrip('/')

        self._index_path = os.path.join(self.workspace_path, "work_index_list.csv.zstd")
        self._output_glob = os.path.join(self.workspace_path, "results", "*.jsonl")

    @staticmethod
    def _compute_workgroup_hash(s3_work_paths: List[str]) -> str:
        """
        Compute a deterministic hash for a group of PDFs.
        
        Args:
            pdfs: List of PDF S3 paths
            
        Returns:
            SHA1 hash of the sorted PDF paths
        """
        sha1 = hashlib.sha1()
        for pdf in sorted(s3_work_paths):
            sha1.update(pdf.encode('utf-8'))
        return sha1.hexdigest()


    async def populate_queue(self, s3_work_paths: str, items_per_group: int) -> None:
        pass

    async def initialize_queue(self) -> None:
        """
        Load the work queue from S3 and initialize it for processing.
        Removes already completed work items and randomizes the order.
        """
        # Load work items and completed items in parallel
        download_task = asyncio.to_thread(
            download_zstd_csv,
            self.s3_client,
            self._index_path
        )
        expand_task = asyncio.to_thread(
            expand_s3_glob,
            self.s3_client,
            self._output_glob
        )
        
        work_queue_lines, done_work_items = await asyncio.gather(download_task, expand_task)

        # Process work queue lines
        work_queue = {
            parts[0]: parts[1:]
            for line in work_queue_lines
            if (parts := line.strip().split(",")) and line.strip()
        }

        # Get set of completed work hashes
        done_work_hashes = {
            os.path.basename(item)[len('output_'):-len('.jsonl')]
            for item in done_work_items
            if os.path.basename(item).startswith('output_') 
            and os.path.basename(item).endswith('.jsonl')
        }

        # Find remaining work and shuffle
        remaining_work_hashes = set(work_queue) - done_work_hashes
        remaining_items = [
            WorkItem(hash_=hash_, pdfs=work_queue[hash_])
            for hash_ in remaining_work_hashes
        ]
        random.shuffle(remaining_items)

        # Initialize queue
        self._queue = asyncio.Queue()
        for item in remaining_items:
            await self._queue.put(item)

        logger.info(f"Initialized queue with {self._queue.qsize()} work items")
    
    async def is_completed(self, work_hash: str) -> bool:
        """
        Check if a work item has been completed.
        
        Args:
            work_hash: Hash of the work item to check
            
        Returns:
            True if the work is completed, False otherwise
        """
        output_s3_path = ""TODO""
        bucket, key = parse_s3_path(output_s3_path)
        
        try:
            await asyncio.to_thread(
                self.s3_client.head_object,
                Bucket=bucket,
                Key=key
            )
            return True
        except self.s3_client.exceptions.ClientError:
            return False

    async def get_work(self) -> Optional[WorkItem]:
        pass        

    def mark_done(self, work_item: WorkItem) -> None:
        """Mark the most recently gotten work item as complete"""
        pass

    @property
    def size(self) -> int:
        """Get current size of work queue"""
        return self._queue.qsize()