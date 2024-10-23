import numpy as np
from io import BytesIO
from PIL import Image
import base64
import random
import torch  # Make sure to import torch as it's used in the DataCollator

from pdelfin.prompts.anchor import get_anchor_text
from pdelfin.prompts import build_finetuning_prompt
from pdelfin.data.renderpdf import render_pdf_to_base64png


def prepare_data_for_qwen2_training(example, processor, target_longest_image_dim: list[int], target_anchor_text_len: list[int]):
    if isinstance(target_longest_image_dim, list): 
        target_longest_image_dim = random.choice(target_longest_image_dim)

    if isinstance(target_anchor_text_len, list): 
        target_anchor_text_len = random.choice(target_anchor_text_len)

    anchor_text = get_anchor_text(example["local_pdf_path"], example["page_num"], pdf_engine="pdfreport", target_length=target_anchor_text_len)
    base64_page_image = render_pdf_to_base64png(example["local_pdf_path"], example["page_num"], target_longest_image_dim=target_longest_image_dim)

    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": base64_page_image
                },
                {"type": "text", "text": build_finetuning_prompt(anchor_text)},
            ],
        }
    ]
    # Apply chat template to get the text
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Decode image from base64
    main_image = Image.open(BytesIO(base64.b64decode(base64_page_image)))

    # Process inputs using processor
    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="np",
    )

    # Get labels by tokenizing the output text
    labels = processor(
        text=[example["response"]],
        padding=True,
        return_tensors="np"
    )
    
    # Append an <|im_end|>\n" to the labels, because this is what it would look like
    # if we passed the whole message stream in there
    im_end_tokens = processor.tokenizer("<|im_end|>\n", add_special_tokens=False)["input_ids"]
    im_end_tokens = np.array(im_end_tokens, dtype=inputs.input_ids.dtype)  # Ensure correct dtype

    # Handle the case where labels['input_ids'] is empty
    if labels['input_ids'].shape[1] == 0:
        labels_input_ids_0 = np.array([], dtype=inputs.input_ids.dtype)
    else:
        labels_input_ids_0 = labels['input_ids'][0].astype(inputs.input_ids.dtype)

    labels['input_ids'] = np.concatenate([labels_input_ids_0, im_end_tokens])
    labels['input_ids'] = np.expand_dims(labels['input_ids'], axis=0)
    
    # Concatenate input_ids and labels
    input_ids = np.concatenate([inputs.input_ids[0], labels.input_ids[0]], axis=0)

    # All columns will participate in attention fully
    attention_mask = np.ones_like(input_ids)

    # Create labels, masking the input portion with -100
    labels_full = np.full_like(input_ids, fill_value=-100)
    labels_full[len(inputs.input_ids[0]):] = labels.input_ids[0]

    # TODO Maybe cap the max length

    # Return as dict, including pixel_values
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels_full,
        "pixel_values": inputs.pixel_values,
        "image_grid_thw": inputs["image_grid_thw"][0]
    }


def batch_prepare_data_for_qwen2_training(batch, processor, target_longest_image_dim: int, target_anchor_text_len: int):
    # Process each example in the batch using the helper function
    processed_examples = []
    for i in range(len(batch["response"])):
        example = {
            "local_pdf_path": batch["local_pdf_path"][i],
            "page_num": batch["page_num"][i],
            "response": batch["response"][i]
        }
        processed_example = prepare_data_for_qwen2_training(example, processor, 
                                                            target_longest_image_dim=target_longest_image_dim,
                                                            target_anchor_text_len=target_anchor_text_len)
        processed_examples.append(processed_example)

    return {
        "input_ids": [x["input_ids"] for x in processed_examples],
        "attention_mask": [x["attention_mask"] for x in processed_examples],
        "labels": [x["labels"] for x in processed_examples],
        "pixel_values": [x["pixel_values"] for x in processed_examples],
        "image_grid_thw": [x["image_grid_thw"] for x in processed_examples],
    }

