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


from pdelfin.train.dataloader import make_dataset
from pdelfin.train.dataprep import batch_prepare_data_for_qwen2_training


def run_train(model_name: str, dataset_path: str):
    if get_rank() == 0:
        logger_level = logging.INFO
    else:
        logger_level = logging.WARN
        disable_progress_bars()

    logger = get_logger(__name__, level=logger_level)
    set_verbosity(logger_level)

    dataset = make_dataset(
        train_data_config=config.train_data,
        valid_data_config=config.valid_data,
        num_proc=config.num_proc,
        logger=logger,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        _attn_implementation="flash_attention_2" if config.model.use_flash_attn else None
    )
    processor = AutoProcessor.from_pretrained(model_name)


    formatted_dataset = dataset.with_transform(partial(batch_prepare_data_for_qwen2_training, processor=processor))
    print(formatted_dataset)
    print("---------------")
    

    with TemporaryDirectory() as output_dir:



        # Uncomment to test speed of data loader
        # train_dataloader = DataLoader(formatted_dataset["train"], batch_size=1, num_workers=4, shuffle=False)
        # for entry in tqdm(train_dataloader):
        #     print("Step!")
        #     model.forward(**{k: v.to("cuda:0") for (k,v) in entry.items()})


def main():
    run_inference(model_name="Qwen/Qwen2-VL-2B-Instruct",
                  dataset_path="s3://ai2-oe-data/jakep/openai_batch_data_v2/*.jsonl")


if __name__ == "__main__":
    main()