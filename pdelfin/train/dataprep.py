import numpy as np
from io import BytesIO
from PIL import Image
import base64
import torch  # Make sure to import torch as it's used in the DataCollator

def prepare_data_for_qwen2_training(example, processor):
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
    labels['input_ids'] = np.concatenate([labels['input_ids'][0], im_end_tokens])
    labels['input_ids'] = np.expand_dims(labels['input_ids'], axis=0)

    # Concatenate input_ids and labels
    input_ids = np.concatenate([inputs.input_ids[0], labels.input_ids[0]], axis=0)
    attention_mask = np.concatenate([inputs.attention_mask[0], labels.attention_mask[0]], axis=0)

    # Create labels, masking the input portion with -100
    labels_full = np.full_like(input_ids, fill_value=-100)
    labels_full[len(inputs.input_ids[0]):] = labels.input_ids[0]

    # Return as dict, including pixel_values
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels_full,
        "pixel_values": inputs.pixel_values[0]
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
