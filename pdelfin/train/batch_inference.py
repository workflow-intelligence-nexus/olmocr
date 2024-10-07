import os
import json
import base64
import logging
import time
from io import BytesIO
from PIL import Image
from functools import partial
from logging import Logger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from tqdm import tqdm

import accelerate
import torch
import torch.distributed
from datasets.utils import disable_progress_bars
from datasets.utils.logging import set_verbosity
from peft import LoraConfig, get_peft_model  # pyright: ignore
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Qwen2VLForConditionalGeneration,
    AutoProcessor
)
from transformers.integrations import WandbCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_utils import get_last_checkpoint
from torch.utils.data import DataLoader

import wandb

from pdelfin.train.core.cli import make_cli, save_config, to_native_types
from pdelfin.train.core.config import TrainConfig
from pdelfin.train.core.loggers import get_logger
from pdelfin.train.core.paths import copy_dir, join_path
from pdelfin.train.core.state import BeakerState

from .utils import (
    RunName,
    get_local_dir,
    log_trainable_parameters,
    packing_collator,
    setup_environment,
)


from pdelfin.train.dataloader import load_jsonl_into_ds, extract_openai_batch_query
from pdelfin.train.dataprep import batch_prepare_data_for_qwen2_inference


@torch.no_grad()
def run_inference(model_name: str, query_dataset_path: str):
    logger = get_logger(__name__, level=logging.INFO)
    set_verbosity(logging.INFO)

    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        _attn_implementation="flash_attention_2",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)

    query_data = load_jsonl_into_ds(query_dataset_path)

    # Map the datasets down to the core fields that we're going to need to make them easier to process
    logger.info("Mapping query data")
    query_data = query_data["train"]
    query_data = query_data.map(extract_openai_batch_query, remove_columns=query_data.column_names)


    formatted_dataset = query_data.with_transform(partial(batch_prepare_data_for_qwen2_inference, processor=processor))
    print(formatted_dataset)
    print("---------------")
    
    
    start_time = None
    toks_generated = 0

    with TemporaryDirectory() as output_dir:
        train_dataloader = DataLoader(formatted_dataset, batch_size=1, num_workers=4, shuffle=False)
        for entry in tqdm(train_dataloader):
            if start_time is None:
                start_time = time.perf_counter()
                
            entry_inputs = {k: v.to("cuda:0") for (k,v) in entry.items()}
            generated_ids = model.generate(**entry_inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(entry_inputs["input_ids"], generated_ids)
            ]

            toks_generated += len(generated_ids_trimmed[0])

            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print(output_text)

            if toks_generated > 2000:
                break

    end_time = time.perf_counter()
    print(f"Tokens/second: {toks_generated / (end_time - start_time):.2f}")


def main():
    run_inference(model_name="Qwen/Qwen2-VL-2B-Instruct",
                  query_dataset_path="s3://ai2-oe-data/jakep/openai_batch_data_v2_mini/*.jsonl")


if __name__ == "__main__":
    main()