import os
import random
import logging
import hashlib
import tempfile
import datetime
from typing import Optional, Tuple, List, Dict, Set
from dataclasses import dataclass
import asyncio
from functools import partial

from olmocr.s3_utils import (
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
        self._queue = asyncio.Queue()

    @staticmethod
    def _compute_workgroup_hash(s3_work_paths: List[str]) -> str:
        """
        Compute a deterministic hash for a group of paths.
        
        Args:
            s3_work_paths: List of S3 paths
            
        Returns:
            SHA1 hash of the sorted paths
        """
        sha1 = hashlib.sha1()
        for path in sorted(s3_work_paths):
            sha1.update(path.encode('utf-8'))
        return sha1.hexdigest()

    async def populate_queue(self, s3_work_paths: list[str], items_per_group: int) -> None:
        """
        Add new items to the work queue.
        
        Args:
            s3_work_paths: Each individual s3 path that we will process over
            items_per_group: Number of items to group together in a single work item
        """
        all_paths = set(s3_work_paths)
        logger.info(f"Found {len(all_paths):,} total paths")

        # Load existing work groups
        existing_lines = await asyncio.to_thread(download_zstd_csv, self.s3_client, self._index_path)
        existing_groups = {}
        for line in existing_lines:
            if line.strip():
                parts = line.strip().split(",")
                group_hash = parts[0]
                group_paths = parts[1:]
                existing_groups[group_hash] = group_paths
                    
        existing_path_set = {path for paths in existing_groups.values() for path in paths}
        
        # Find new paths to process
        new_paths = all_paths - existing_path_set
        logger.info(f"{len(new_paths):,} new paths to add to the workspace")

        if not new_paths:
            return

        # Create new work groups
        new_groups = []
        current_group = []
        for path in sorted(new_paths):
            current_group.append(path)
            if len(current_group) == items_per_group:
                group_hash = self._compute_workgroup_hash(current_group)
                new_groups.append((group_hash, current_group))
                current_group = []
        if current_group:
            group_hash = self._compute_workgroup_hash(current_group)
            new_groups.append((group_hash, current_group))

        logger.info(f"Created {len(new_groups):,} new work groups")

        # Combine and save updated work groups
        combined_groups = existing_groups.copy()
        for group_hash, group_paths in new_groups:
            combined_groups[group_hash] = group_paths

        combined_lines = [
            ",".join([group_hash] + group_paths)
            for group_hash, group_paths in combined_groups.items()
        ]

        if new_groups:
            await asyncio.to_thread(
                upload_zstd_csv,
                self.s3_client,
                self._index_path,
                combined_lines
            )

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
            WorkItem(hash=hash_, s3_work_paths=work_queue[hash_])
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
        output_s3_path = os.path.join(self.workspace_path, "results", f"output_{work_hash}.jsonl")
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

    async def get_work(self, worker_lock_timeout_secs: int = 1800) -> Optional[WorkItem]:
        """
        Get the next available work item that isn't completed or locked.
        
        Args:
            worker_lock_timeout_secs: Number of seconds before considering a worker lock stale (default 30 mins)
            
        Returns:
            WorkItem if work is available, None if queue is empty
        """
        while True:
            try:
                work_item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return None

            # Check if work is already completed
            if await self.is_completed(work_item.hash):
                logger.debug(f"Work item {work_item.hash} already completed, skipping")
                self._queue.task_done()
                continue

            # Check for worker lock
            lock_path = os.path.join(self.workspace_path, "worker_locks", f"output_{work_item.hash}.jsonl")
            bucket, key = parse_s3_path(lock_path)
            
            try:
                response = await asyncio.to_thread(
                    self.s3_client.head_object,
                    Bucket=bucket,
                    Key=key
                )
                
                # Check if lock is stale
                last_modified = response['LastModified']
                if (datetime.datetime.now(datetime.timezone.utc) - last_modified).total_seconds() > worker_lock_timeout_secs:
                    # Lock is stale, we can take this work
                    logger.debug(f"Found stale lock for {work_item.hash}, taking work item")
                else:
                    # Lock is active, skip this work
                    logger.debug(f"Work item {work_item.hash} is locked by another worker, skipping")
                    self._queue.task_done()
                    continue
                    
            except self.s3_client.exceptions.ClientError:
                # No lock exists, we can take this work
                pass

            # Create our lock file
            try:
                await asyncio.to_thread(
                    self.s3_client.put_object,
                    Bucket=bucket,
                    Key=key,
                    Body=b''
                )
            except Exception as e:
                logger.warning(f"Failed to create lock file for {work_item.hash}: {e}")
                self._queue.task_done()
                continue

            return work_item

    async def mark_done(self, work_item: WorkItem) -> None:
        """
        Mark a work item as done by removing its lock file.
        
        Args:
            work_item: The WorkItem to mark as done
        """
        lock_path = os.path.join(self.workspace_path, "worker_locks", f"output_{work_item.hash}.jsonl")
        bucket, key = parse_s3_path(lock_path)
        
        try:
            await asyncio.to_thread(
                self.s3_client.delete_object,
                Bucket=bucket,
                Key=key
            )
        except Exception as e:
            logger.warning(f"Failed to delete lock file for {work_item.hash}: {e}")

        self._queue.task_done()

    @property
    def size(self) -> int:
        """Get current size of work queue"""
        return self._queue.qsize()