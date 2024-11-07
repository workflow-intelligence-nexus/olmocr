import logging
import argparse
import boto3
import os

from tqdm import tqdm

from pdelfin.s3_utils import expand_s3_glob, parse_s3_path, download_zstd_csv, upload_zstd_csv, download_directory

# Basic logging setup for now
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

# Quiet logs from pypdf
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Global s3 client for the whole script, feel free to adjust params if you need it
workspace_s3 = boto3.client('s3')
pdf_s3 = boto3.client('s3')


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

    parser.add_argument('--model', help='List of paths where you can find the model to convert this pdf. You can specify several different paths here, and the script will try to use the one which is fastest to access',
                         default=["weka://oe-data-default/jakep/Qwen_Qwen2-VL-7B-Instruct-e4ecf8-01JAH8GMWHTJ376S2N7ETXRXH4/best_bf16/",
                                  "gs://ai2-oe-data/jakep/experiments/qwen2vl-pdf/v1/models/jakep/Qwen_Qwen2-VL-7B-Instruct-e4ecf8-01JAH8GMWHTJ376S2N7ETXRXH4/checkpoint-9500/bf16/",
                                  "s3://ai2-oe-data/jakep/experiments/qwen2vl-pdf/v1/models/jakep/Qwen_Qwen2-VL-7B-Instruct-e4ecf8-01JAH8GMWHTJ376S2N7ETXRXH4/checkpoint-9500/bf16/"])
    parser.add_argument('--model_max_context', type=int, default="8192", help="Maximum context length that the model was fine tuned under")
    parser.add_argument('--model_chat_template', type=str, default="qwen2-vl", help="Chat template to pass to sglang server")
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
        if new_groups:
            upload_zstd_csv(workspace_s3, index_file_s3_path, combined_lines)

        logger.info("Completed adding new PDFs.")

    # TODO
    # If there is a beaker flag, then your job is to trigger this script with N replicas on beaker
    # If not, then your job is to do the actual work

    # Donwload the model from the best place available
    model_cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'pdelfin', 'model')
    download_directory(args.model, model_cache_dir)

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
