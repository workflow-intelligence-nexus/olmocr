import argparse
import os
import json
from smart_open import smart_open

def main():
    parser = argparse.ArgumentParser(description='Fix up a Qwen2VL checkpoint saved on s3 or otherwise, so that it will load properly in vllm/birr')
    parser.add_argument('s3_path', type=str, help='S3 path to the Hugging Face checkpoint.')
    args = parser.parse_args()

    qwen_replacement_files = [
        # Config is special to fix rope config
        "s3://ai2-oe-data/artifacts/Qwen2-VL-7B-Instruct/config.json",

        # Tokenizer and preprocessor are just not saved in the usual flow
        "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/tokenizer.json",
        "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/tokenizer_config.json",
        "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/vocab.json",
        "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/merges.txt",
        "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/generation_config.json",
        "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/chat_template.json",
        "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/preprocessor_config.json",
    ]

    # Now, download the config.json from the original path and verify the architectures
    config_path = os.path.join(args.s3_path, "config.json")

    with smart_open(config_path, 'r') as f:
        config_data = json.load(f)

    assert config_data["architectures"] == ["Qwen2VLForConditionalGeneration"]

    # Iterate over each file in the replacement list
    for replacement_file in qwen_replacement_files:
        filename = os.path.basename(replacement_file)
        dest_path = os.path.join(args.s3_path, filename)
        
        with smart_open(replacement_file, 'rb') as src_file:
            data = src_file.read()

        with smart_open(dest_path, 'wb') as dest_file:
            dest_file.write(data)

    print("Model updated successfully.")

if __name__ == '__main__':
    main()
