import unittest
import base64
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor

from pdelfin.train.dataloader import (
    build_batch_query_response_vision_dataset,
)

from pdelfin.train.dataprep import (
    prepare_data_for_qwen2_training
)


class TestDataprep(unittest.TestCase):
    def testTokenizationMatches(self):
        ds = build_batch_query_response_vision_dataset(
            query_glob_path="s3://ai2-oe-data/jakep/openai_batch_data_v2_mini/*.jsonl",
            response_glob_path="s3://ai2-oe-data/jakep/openai_batch_done_v2_mini/*.json",
        )

        example = ds[0]

        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        full_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": example["input_prompt_image_base64"]  # Placeholder
                    },
                    {"type": "text", "text": example["input_prompt_text"]},
                ],
            },

            {
                "role": "assistant",
                "content": example["response"]
            }
        ]

        text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=True)

        # Decode image from base64
        main_image = Image.open(BytesIO(base64.b64decode(example["input_prompt_image_base64"])))

        # Process inputs using processor
        inference_inputs = processor(
            text=[text],
            images=[main_image],
            padding=True,
            return_tensors="np",
        )

        print(inference_inputs)
        print(inference_inputs["input_ids"].shape)

        training_inputs = prepare_data_for_qwen2_training(example, processor=processor)

        print(training_inputs)
        print(training_inputs["input_ids"].shape)

