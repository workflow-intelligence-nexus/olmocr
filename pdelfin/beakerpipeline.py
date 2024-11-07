import argparse
import subprocess
import signal
import sys
import os
import time
import tempfile
import redis
import redis.exceptions
import random
import boto3
import atexit

from pdelfin.s3_utils import expand_s3_glob

workspace_s3 = boto3.client('s3')
pdf_s3 = boto3.client('s3')

LOCK_KEY = "queue_populating"
LOCK_TIMEOUT = 30  # seconds

def populate_queue_if_empty(queue, s3_glob_path, redis_client):
    """
    Check if the queue is empty. If it is, attempt to acquire a lock to populate it.
    Only one worker should populate the queue at a time.
    """
    if queue.llen("work_queue") == 0:
        # Attempt to acquire the lock
        lock_acquired = redis_client.set(LOCK_KEY, "locked", nx=True, ex=LOCK_TIMEOUT)
        if lock_acquired:
            print("Acquired lock to populate the queue.")
            try:
                paths = expand_s3_glob(pdf_s3, s3_glob_path)
                if not paths:
                    print("No paths found to populate the queue.")
                    return
                for path in paths:
                    queue.rpush("work_queue", path)
                print("Queue populated with initial work items.")
            except Exception as e:
                print(f"Error populating queue: {e}")
                # Optionally, handle retry logic or alerting here
            finally:
                # Release the lock
                redis_client.delete(LOCK_KEY)
                print("Released lock after populating the queue.")
        else:
            print("Another worker is populating the queue. Waiting for it to complete.")
            # Optionally, wait until the queue is populated
            wait_for_queue_population(queue)

def wait_for_queue_population(queue, wait_time=5, max_wait=60):
    """
    Wait until the queue is populated by another worker.
    """
    elapsed = 0
    while elapsed < max_wait:
        queue_length = queue.llen("work_queue")
        if queue_length > 0:
            print("Queue has been populated by another worker.")
            return
        print(f"Waiting for queue to be populated... ({elapsed + wait_time}/{max_wait} seconds)")
        time.sleep(wait_time)
        elapsed += wait_time
    print("Timeout waiting for queue to be populated.")
    sys.exit(1)

def process(item):
    # Simulate processing time between 1 and 3 seconds
    print(f"Processing item: {item}")
    time.sleep(0.5)
    print(f"Completed processing item: {item}")

def get_redis_client(sentinel, master_name, leader_ip, leader_port, max_wait=60):
    """
    Obtain a Redis client using Sentinel, with retry logic.
    """
    elapsed = 0
    wait_interval = 1  # seconds
    while elapsed < max_wait:
        try:
            r = sentinel.master_for(master_name, socket_timeout=0.1, decode_responses=True)
            r.ping()
            print(f"Connected to Redis master at {leader_ip}:{leader_port}")
            return r
        except redis.exceptions.ConnectionError as e:
            print(f"Attempt {elapsed + 1}: Unable to connect to Redis master at {leader_ip}:{leader_port}. Retrying in {wait_interval} second(s)...")
            time.sleep(wait_interval)
            elapsed += wait_interval
    print(f"Failed to connect to Redis master at {leader_ip}:{leader_port} after {max_wait} seconds. Exiting.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Set up Redis Sentinel-based worker queue.')
    parser.add_argument('--leader-ip', help='IP address of the initial leader node')
    parser.add_argument('--leader-port', type=int, default=6379, help='Port of the initial leader node')
    parser.add_argument('--replica', type=int, required=True, help='Replica number (0 to N-1)')
    parser.add_argument('--add-pdfs', help='S3 glob path for work items')

    args = parser.parse_args()

    replica_number = args.replica

    base_redis_port = 6379
    base_sentinel_port = 26379

    redis_port = base_redis_port + replica_number
    sentinel_port = base_sentinel_port + replica_number

    if replica_number == 0:
        leader_ip = args.leader_ip if args.leader_ip else '127.0.0.1'
        leader_port = args.leader_port
    else:
        if not args.leader_ip:
            print('Error: --leader-ip is required for replica nodes (replica_number >= 1)')
            sys.exit(1)
        leader_ip = args.leader_ip
        leader_port = args.leader_port

    temp_dir = tempfile.mkdtemp()
    redis_conf_path = os.path.join(temp_dir, 'redis.conf')
    sentinel_conf_path = os.path.join(temp_dir, 'sentinel.conf')

    print("Redis config path:", redis_conf_path)

    with open(redis_conf_path, 'w') as f:
        f.write(f'port {redis_port}\n')
        f.write(f'dbfilename dump-{replica_number}.rdb\n')
        f.write(f'appendfilename "appendonly-{replica_number}.aof"\n')
        f.write(f'logfile "redis-{replica_number}.log"\n')
        f.write(f'dir {temp_dir}\n')
        if replica_number == 0:
            f.write('bind 0.0.0.0\n')
        else:
            f.write(f'replicaof {leader_ip} {leader_port}\n')

    master_name = 'mymaster'
    quorum = 1

    with open(sentinel_conf_path, 'w') as f:
        f.write(f'port {sentinel_port}\n')
        f.write(f'dir {temp_dir}\n')
        f.write(f'sentinel monitor {master_name} {leader_ip} {leader_port} {quorum}\n')
        f.write(f'sentinel down-after-milliseconds {master_name} 5000\n')
        f.write(f'sentinel failover-timeout {master_name} 10000\n')
        f.write(f'sentinel parallel-syncs {master_name} 1\n')

    redis_process = subprocess.Popen(['redis-server', redis_conf_path])
    sentinel_process = subprocess.Popen(['redis-sentinel', sentinel_conf_path])

    # Register atexit function to guarantee process termination
    def terminate_processes():
        print("Terminating child processes...")
        redis_process.terminate()
        sentinel_process.terminate()
        try:
            redis_process.wait(timeout=5)
            sentinel_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Forcing termination of child processes.")
            redis_process.kill()
            sentinel_process.kill()
        print("Child processes terminated.")

    atexit.register(terminate_processes)

    # Also handle signal-based termination
    def handle_signal(signum, frame):
        print(f"Received signal {signum}. Terminating processes...")
        terminate_processes()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    time.sleep(2)

    # Use Sentinel to connect to the master
    from redis.sentinel import Sentinel
    sentinel = Sentinel([('127.0.0.1', sentinel_port)], socket_timeout=0.1)

    # Initial connection to Redis master
    redis_client = get_redis_client(sentinel, master_name, leader_ip, leader_port)

    # Populate the work queue if it's empty, using a distributed lock
    populate_queue_if_empty(redis_client, args.add_pdfs, redis_client)

    try:
        while True:
            try:
                # Try to get an item from the queue with a 1-minute timeout for processing
                work_item = redis_client.brpoplpush("work_queue", "processing_queue", 60)
                if work_item:
                    try:
                        process(work_item)
                        # Remove from the processing queue if processed successfully
                        redis_client.lrem("processing_queue", 1, work_item)
                    except Exception as e:
                        print(f"Error processing {work_item}: {e}")
                        # If an error occurs, let it be requeued after timeout

                queue_length = redis_client.llen("work_queue")
                print(f"Total work items in queue: {queue_length}")

                time.sleep(0.1)

            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
                print("Lost connection to Redis. Attempting to reconnect using Sentinel...")
                # Attempt to reconnect using Sentinel
                while True:
                    try:
                        redis_client = get_redis_client(sentinel, master_name, leader_ip, leader_port)
                        print("Reconnected to Redis master.")
                        break  # Exit the reconnection loop and resume work
                    except redis.exceptions.ConnectionError:
                        print("Reconnection failed. Retrying in 5 seconds...")
                        time.sleep(5)
            except Exception as e:
                print(f"Unexpected error: {e}")
                handle_signal(None, None)

    except KeyboardInterrupt:
        handle_signal(None, None)

if __name__ == '__main__':
    main()
