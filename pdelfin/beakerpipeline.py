import logging
import argparse
import boto3
import os

from tqdm import tqdm
from urllib.parse import urlparse
import zstandard as zstd
from io import BytesIO, TextIOWrapper

from pdelfin.s3_utils import expand_s3_glob, get_s3_bytes, parse_s3_path, put_s3_bytes

# Basic logging setup for now
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

# Quiet logs from pypdf
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Global s3 client for the whole script, feel free to adjust params if you need it
workspace_s3 = boto3.client('s3')
pdf_s3 = boto3.client('s3')


def download_zstd_csv(s3_client, s3_path):
    """Download and decompress a .zstd CSV file from S3."""
    try:
        compressed_data = get_s3_bytes(s3_client, s3_path)
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(compressed_data)
        text_stream = TextIOWrapper(BytesIO(decompressed), encoding='utf-8')
        lines = text_stream.readlines()
        logger.info(f"Downloaded and decompressed {s3_path}")
        return lines
    except s3_client.exceptions.NoSuchKey:
        logger.info(f"No existing {s3_path} found in s3, starting fresh.")
        return []


def upload_zstd_csv(s3_client, s3_path, lines):
    """Compress and upload a list of lines as a .zstd CSV file to S3."""
    joined_text = "\n".join(lines)
    compressor = zstd.ZstdCompressor()
    compressed = compressor.compress(joined_text.encode('utf-8'))
    put_s3_bytes(s3_client, s3_path, compressed)
    logger.info(f"Uploaded compressed {s3_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manager for running millions of PDFs through a batch inference pipeline')
    parser.add_argument('workspace', help='The S3 path where work will be done e.g., s3://bucket/prefix/')
    parser.add_argument('--pdfs', help='Path to add pdfs stored in s3 to the workspace, can be a glob path s3://bucket/prefix/*.pdf or path to file containing list of pdf paths', default=None)
    parser.add_argument('--target_longest_image_dim', type=int, help='Dimension on longest side to use for rendering the pdf pages', default=1024)
    parser.add_argument('--target_anchor_text_len', type=int, help='Maximum amount of anchor text to use (characters)', default=6000)
    parser.add_argument('--workspace_profile', help='S3 configuration profile for accessing the workspace', default=None)
    parser.add_argument('--pdf_profile', help='S3 configuration profile for accessing the raw pdf documents', default=None)
    parser.add_argument('--group_size', type=int, default=20, help='Number of pdfs that will be part of each work item in the work queue.')
    parser.add_argument('--workers', type=int, default=10, help='Number of workers to run at a time')

    args = parser.parse_args()

    if args.workspace_profile:
        workspace_session = boto3.Session(profile_name=args.workspace_profile)
        workspace_s3 = workspace_session.client("s3")

    if args.pdf_profile:
        pdf_session = boto3.Session(profile_name=args.pdf_profile)
        pdf_s3 = pdf_session.client("s3")

    # Check list of pdfs and that it matches what's in the workspace
    if args.pdfs:
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
        
        index_file_s3_path = os.path.join(args.workspace, "pdf_index_list.csv.zstd")
        existing_lines = download_zstd_csv(workspace_s3, index_file_s3_path)

        # Parse existing work items into groups
        existing_groups = [line.strip().split(",") for line in existing_lines if line.strip()]
        existing_pdf_set = set(pdf for group in existing_groups for pdf in group)

        logger.info(f"Loaded {len(existing_pdf_set):,} existing pdf paths from the workspace")

        # Remove existing PDFs from all_pdfs
        new_pdfs = all_pdfs - existing_pdf_set
        logger.info(f"{len(new_pdfs):,} new pdf paths to add to the workspace")

        # Group the new PDFs into chunks of group_size
        new_groups = []
        current_group = []
        for pdf in sorted(new_pdfs):  # Sort for consistency
            current_group.append(pdf)
            if len(current_group) == args.group_size:
                new_groups.append(current_group)
                current_group = []
        if current_group:
            new_groups.append(current_group)

        logger.info(f"Created {len(new_groups):,} new work groups")

        # Combine existing groups with new groups
        combined_groups = existing_groups + new_groups

        # Prepare lines to write back
        combined_lines = [",".join(group) for group in combined_groups]

        # Upload the combined work items back to S3
        upload_zstd_csv(workspace_s3, index_file_s3_path, combined_lines)

        logger.info("Completed adding new PDFs.")


    # If there is a beaker flag, then your job is to trigger this script with N replicas on beaker
    # If not, then your job is to do the actual work

    # Start up the sglang server

    # Read in the work queue from s3
    # Read in the done items from the s3 workspace

    # Spawn up to N workers to do:
        # In a loop, take a random work item, read in the pdfs, queue in their requests
        # Get results back, retry any failed pages
        # Check periodically if that work is done in s3, if so, then abandon this work
        # Save results back to s3 workspace output folder

    # Possible future addon, in beaker, discover other nodes on this same job
    # Send them a message when you take a work item off the queue
