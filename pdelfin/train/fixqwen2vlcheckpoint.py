import argparse
import os
import tempfile
import boto3
from tqdm import tqdm
from transformers import AutoModel, Qwen2VLForConditionalGeneration
from smart_open import smart_open


def main():
    parser = argparse.ArgumentParser(description='Fix up a Qwen2VL checkpoint saved on s3 or otherwise, so that it will load properly in vllm/birr')
    parser.add_argument('s3_path', type=str, help='S3 path to the Hugging Face checkpoint.')
    args = parser.parse_args()

    # Create a temporary directory to store the model files
 
    # Rewrite the config.json from the official repo, this fixes a weird bug with the rope scaling configuration
    with smart_open("https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/config.json", "r") as newf:
        new_config = newf.read()

    with smart_open(os.path.join(args.s3_path, "config.json"), "w") as oldf:
        oldf.write(new_config)

    print("Model updated successfully.")

if __name__ == '__main__':
    main()
