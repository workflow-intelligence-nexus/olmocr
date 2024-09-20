# Step 1, load the data
# Probably, we want to see just a folder with openai batch input jsonls, plus the batch output jsonls
# TODO: Figure out hyperparameters for image sizing
# Step 2. Load those prompts through and do a forward pass to calculate the loss

# Step 3. Add hugging face accelerate for training

# Step 4. Checkpointing code, both saving and reloading to restart

# Step 5. Move over from interactive session to gantry launch script

import os
import json
import base64
import logging
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


from pdelfin.train.dataloader import build_batch_query_response_vision_dataset
from pdelfin.train.dataprep import batch_prepare_data_for_qwen2_training


def run_train(config: TrainConfig):
    train_ds = build_batch_query_response_vision_dataset(
                        query_glob_path="s3://ai2-oe-data/jakep/openai_batch_data_v2_mini/*.jsonl",
                        response_glob_path="s3://ai2-oe-data/jakep/openai_batch_done_v2_mini/*.json",
                    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    train_ds = train_ds.with_transform(partial(batch_prepare_data_for_qwen2_training, processor=processor))
    print(train_ds)
    
    dataloader = DataLoader(train_ds, batch_size=1, shuffle=False)

    for batch in dataloader:
        print(batch)

        result = model.forward(**batch)



def main():
    train_config = make_cli(TrainConfig)  # pyright: ignore
    run_train(train_config)


if __name__ == "__main__":
    main()