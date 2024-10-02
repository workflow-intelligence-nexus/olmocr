import json
import multiprocessing
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha1
from logging import Logger
from tempfile import TemporaryDirectory
from typing import Dict, Generator, List, Optional, TypeVar

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import PrecisionType

from .core.cli import to_native_types
from .core.config import AwsConfig, TrainConfig, WandbConfig
from .core.loggers import get_logger
from .core.paths import copy_dir, is_local
from .core.state import BeakerState
#from .tokenization import ModelTokenizer

T = TypeVar("T")


def accelerator_to_dtype(accelerator: Accelerator) -> torch.dtype:
    pt = PrecisionType(accelerator.mixed_precision)
    if pt == PrecisionType.FP16:
        return torch.float16
    elif pt == PrecisionType.BF16:
        return torch.bfloat16
    elif pt == PrecisionType.FP8:
        return torch.float8_e4m3fn
    return torch.float32


def setup_environment(
    aws_config: Optional[AwsConfig] = None, wandb_config: Optional[WandbConfig] = None, **kwargs: str
):
    multiprocessing.set_start_method("spawn", force=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "false"

    if wandb_config:
        os.environ["WANDB_WATCH"] = "false"

    for key, value in to_native_types(wandb_config or {}).items():
        if value is not None:
            os.environ[f"WANDB_{key.upper()}"] = str(value)

    for key, value in to_native_types(aws_config or {}).items():
        if value is not None:
            os.environ[f"AWS_{key.upper()}"] = str(value)

    os.environ.update(kwargs)


@dataclass
class RunName:
    run: str
    group: str

    @classmethod
    def get(cls, config: TrainConfig, accelerator: Optional[Accelerator] = None) -> "RunName":
        job_rank = f"-{accelerator.process_index}" if accelerator else ""

        if beaker_job_id := BeakerState().job_id:
            job_id = f"-{beaker_job_id}"
        else:
            job_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        (config_hash := sha1()).update(json.dumps(to_native_types(config)).encode())
        model_name = config.model.name_or_path.replace("/", "_")

        group_name = f"{model_name}-{config_hash.hexdigest()[:6]}"
        run_name = f"{group_name}{job_id}{job_rank}"
        return cls(group=group_name, run=run_name)


@contextmanager
def override_torch_threads(n: int):
    torch_num_threads = torch.get_num_threads()
    torch.set_num_threads(n)

    yield

    torch.set_num_threads(torch_num_threads)


@contextmanager
def temp_args(obj: T, **kwargs) -> Generator[T, None, None]:
    orig = {k: getattr(obj, k) for k in kwargs.keys()}
    for k, v in kwargs.items():
        setattr(obj, k, v)

    yield obj

    for k, v in orig.items():
        setattr(obj, k, v)


def log_trainable_parameters(model: torch.nn.Module, logger: Optional[Logger] = None):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    (logger or get_logger(__name__)).info(
        "trainable params: %s || all params: %s || trainable%%: %s",
        f"{trainable_params:,}",
        f"{all_param:,}",
        f"{trainable_params / all_param:.2%}",
    )


def packing_collator(batch, pad_multiple_of: int, do_shrink: bool = True):
    with override_torch_threads(1):
        # start by stacking
        stacked_batch = {k: torch.tensor([s[k] for s in batch]) for k in batch[0].keys()}

        if not do_shrink:
            return stacked_batch

        # find first position where attention mask is 0 for all samples
        max_pos = int(stacked_batch["attention_mask"].sum(0).argmin())
        max_pos_multiple_of = max_pos + (pad_multiple_of - max_pos % pad_multiple_of)

        if max_pos_multiple_of >= len(batch[0]["attention_mask"]):
            # no need to crop
            return stacked_batch

        # crop the tensors
        cropped_batch = {k: v[:, :max_pos_multiple_of] for k, v in stacked_batch.items()}

    return cropped_batch


@contextmanager
def get_local_dir(output_dir: str):

    with TemporaryDirectory() as tmp_dir:
        if is_local(output_dir):
            yield output_dir
        else:
            yield tmp_dir
            copy_dir(tmp_dir, output_dir)

            