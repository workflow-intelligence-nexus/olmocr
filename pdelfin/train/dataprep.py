import numpy as np
from io import BytesIO
from PIL import Image
import base64
import torch  # Make sure to import torch as it's used in the DataCollator

from pdelfin.prompts import build_finetuning_prompt

def filter_by_max_seq_len(example, processor, max_prompt_len: int=2200, max_response_len: int=2200):
    if len(processor.tokenizer.tokenize(example["input_prompt_text"])) > max_prompt_len:
        return False
    
    if len(processor.tokenizer.tokenize(example["response"])) > max_response_len:
        return False
    
    return True


def prepare_data_for_qwen2_training(example, processor, add_batch_dim=False):
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": example["input_prompt_image_base64"]  # Placeholder
                },
                {"type": "text", "text": build_finetuning_prompt(example["raw_page_text"])},
            ],
        }
    ]
    # Apply chat template to get the text
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Decode image from base64
    main_image = Image.open(BytesIO(base64.b64decode(example["input_prompt_image_base64"])))

    # Right now, we are going to downsample to 1024 on the longest dimension, because
    # 2048 as we passed to OpenAI is too large for training
    width, height = main_image.size
    assert 1800 <= max(width, height) <= 2200, f"Image size {width}x{height} invalid"
    main_image = main_image.resize((width // 2, height // 2), Image.LANCZOS)


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

    print(labels["input_ids"].shape)
    
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

    # Return as dict, including pixel_values
    if add_batch_dim:
      return {
            "input_ids": input_ids[np.newaxis, ...],
            "attention_mask": attention_mask[np.newaxis, ...],
            "labels": labels_full[np.newaxis, ...],
            "pixel_values": inputs.pixel_values[np.newaxis, ...],
            "image_grid_thw": inputs["image_grid_thw"]
        }
    else:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_full,
            "pixel_values": inputs.pixel_values,
            "image_grid_thw": inputs["image_grid_thw"][0]
        }


def batch_prepare_data_for_qwen2_training(batch, processor):
    # Process each example in the batch using the helper function
    processed_examples = []
    for i in range(len(batch["input_prompt_image_base64"])):
        example = {
            "input_prompt_image_base64": batch["input_prompt_image_base64"][i],
            "input_prompt_text": batch["input_prompt_text"][i],
            "raw_page_text": batch["raw_page_text"][i],
            "response": batch["response"][i]
        }
        processed_example = prepare_data_for_qwen2_training(example, processor)
        processed_examples.append(processed_example)

    return {
        "input_ids": [x["input_ids"] for x in processed_examples],
        "attention_mask": [x["attention_mask"] for x in processed_examples],
        "labels": [x["labels"] for x in processed_examples],
        "pixel_values": [x["pixel_values"] for x in processed_examples],
        "image_grid_thw": [x["image_grid_thw"] for x in processed_examples],
    }


def prepare_data_for_qwen2_inference(example, processor):
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": example["input_prompt_image_base64"]  # Placeholder
                },
                {"type": "text", "text": example["input_prompt_text"]},
            ],
        }
    ]
    # Apply chat template to get the text
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Decode image from base64
    main_image = Image.open(BytesIO(base64.b64decode(example["input_prompt_image_base64"])))

    # Right now, we are going to downsample to 1024 on the longest dimension, because
    # 2048 as we passed to OpenAI is too large for training
    width, height = main_image.size
    assert 1800 <= max(width, height) <= 2200
    main_image = main_image.resize((width // 2, height // 2), Image.LANCZOS)


    # Process inputs using processor
    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="np",
    )

    input_ids = inputs["input_ids"][0]

    # All columns will participate in attention fully
    attention_mask = np.ones_like(input_ids)

    # Return as dict, including pixel_values
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": inputs.pixel_values,
        "image_grid_thw": inputs["image_grid_thw"][0]
    }


def batch_prepare_data_for_qwen2_inference(batch, processor):
    # Process each example in the batch using the helper function
    processed_examples = []
    for i in range(len(batch["input_prompt_image_base64"])):
        example = {
            "input_prompt_image_base64": batch["input_prompt_image_base64"][i],
            "input_prompt_text": batch["input_prompt_text"][i],
            "raw_page_text": batch["raw_page_text"][i],
        }
        processed_example = prepare_data_for_qwen2_inference(example, processor)
        processed_examples.append(processed_example)

    return {
        "input_ids": [x["input_ids"] for x in processed_examples],
        "attention_mask": [x["attention_mask"] for x in processed_examples],
        "pixel_values": [x["pixel_values"] for x in processed_examples],
        "image_grid_thw": [x["image_grid_thw"] for x in processed_examples],
    }

# Define a custom data collator
class DataCollatorForVisionLanguageModeling:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_ids = [f['input_ids'] for f in features]
        attention_mask = [f['attention_mask'] for f in features]
        labels = [f['labels'] for f in features]
        pixel_values = [f['pixel_values'] for f in features]

        # Pad input_ids, attention_mask, labels
        batch = self.processor.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels},
            return_tensors="pt",
            padding=True,
        )

        # Stack pixel_values
        batch['pixel_values'] = torch.stack([torch.tensor(pv) for pv in pixel_values])

        return batch
