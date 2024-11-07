import argparse
import subprocess
import signal
import sys
import os
import time
import tempfile
import redis
import random
import boto3
import atexit

from pdelfin.s3_utils import expand_s3_glob

workspace_s3 = boto3.client('s3')
pdf_s3 = boto3.client('s3')

def populate_queue_if_empty(queue, s3_glob_path):
    # Check if the queue is empty, and if so, populate it with work items
    if queue.llen("work_queue") == 0:
        paths = expand_s3_glob(pdf_s3, s3_glob_path)
        for path in paths:
            queue.rpush("work_queue", path)
        print("Queue populated with initial work items.")

def process(item):
    # Simulate processing time between 1 and 3 seconds
    print(f"Processing item: {item}")
    time.sleep(random.randint(1, 3))
    print(f"Completed processing item: {item}")

def main():
    parser = argparse.ArgumentParser(description='Set up Redis Sentinel-based worker queue.')
    parser.add_argument('--leader-ip', help='IP address of the initial leader node')
    parser.add_argument('--leader-port', type=int, default=6379, help='Port of the initial leader node')
    parser.add_argument('--replica', type=int, required=True, help='Replica number (0 to N-1)')
    parser.add_argument('--add-pdfs', required=True, help='S3 glob path for work items')

    args = parser.parse_args()

    replica_number = args.replica

    base_redis_port = 6379
    base_sentinel_port = 26379

    redis_port = base_redis_port + replica_number
    # Set sentinel_port to be the same on all nodes
    sentinel_port = base_sentinel_port

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
    quorum = 2

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
        redis_process.terminate()
        sentinel_process.terminate()
        redis_process.wait()  # Ensures subprocess is cleaned up
        sentinel_process.wait()
        print("Child processes terminated.")

    atexit.register(terminate_processes)

    # Also handle signal-based termination
    def handle_signal(signum, frame):
        terminate_processes()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    time.sleep(2)

    # Use Sentinel to connect to the master
    from redis.sentinel import Sentinel
    sentinel = Sentinel([('127.0.0.1', sentinel_port)], socket_timeout=0.1)
    r = sentinel.master_for(master_name, socket_timeout=0.1, decode_responses=True)

    # Populate the work queue if this is the leader (replica 0)
    if replica_number == 0:
        populate_queue_if_empty(r, args.add_pdfs)

    try:
        while True:
            # Try to get an item from the queue with a 1-minute timeout for processing
            work_item = r.brpoplpush("work_queue", "processing_queue", 60)
            if work_item:
                try:
                    process(work_item)
                    # Remove from the processing queue if processed successfully
                    r.lrem("processing_queue", 1, work_item)
                except Exception as e:
                    print(f"Error processing {work_item}: {e}")
                    # If an error occurs, let it be requeued after timeout

            queue_length = r.llen("work_queue")
            print(f"Total work items in queue: {queue_length}")
            time.sleep(1)
    except KeyboardInterrupt:
        handle_signal(None, None)

if __name__ == '__main__':
    main()
