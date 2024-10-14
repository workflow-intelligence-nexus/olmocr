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
    AutoProcessor,
    Qwen2VLConfig
)


from pdelfin.data.renderpdf import render_pdf_to_base64png
from pdelfin.prompts.anchor import get_anchor_text
from pdelfin.prompts.prompts import build_finetuning_prompt

from pdelfin.train.dataprep import prepare_data_for_qwen2_inference

def build_page_query(local_pdf_path: str, page: int) -> dict:
    image_base64 = render_pdf_to_base64png(local_pdf_path, page, 1024)
    anchor_text = get_anchor_text(local_pdf_path, page, pdf_engine="pdfreport")

    return {
        "input_prompt_text": build_finetuning_prompt(anchor_text),
        "input_prompt_image_base64": image_base64
    }


@torch.no_grad()
def run_inference(model_name: str):    
    config = Qwen2VLConfig.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)

    # If it doesn't load, change the type:mrope key to "default"

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, device_map="auto", config=config)
    model.eval()
  

    query = build_page_query(os.path.join(os.path.dirname(__file__), "..", "..", "tests", "gnarly_pdfs", "overrun_on_pg8.pdf"), 8)

    inputs = prepare_data_for_qwen2_inference(query, processor)

    print(inputs)

    inputs = {
        x: torch.from_numpy(y).unsqueeze(0).to("cuda")
            for (x,y) in inputs.items()
    }

    output_ids = model.generate(**inputs, temperature=0.8, do_sample=True, max_new_tokens=1500)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs["input_ids"], output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    print(output_text)



def main():
    run_inference(model_name="/root/model")


if __name__ == "__main__":
    main()