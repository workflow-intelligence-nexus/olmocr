import unittest
import base64
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor

from pdelfin.train.dataloader import (
    build_batch_query_response_vision_dataset,
)

from pdelfin.train.dataprep import (
    prepare_data_for_qwen2_training, build_finetuning_prompt
)
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from pdelfin.train.utils import make_dataset
from pdelfin.train.core.config import TrainConfig, DataConfig, SourceConfig

class TestDataprep(unittest.TestCase):
    def testFullDataloader(self):
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        config = TrainConfig(
            train_data=DataConfig(seed=42,
                                  sources=[SourceConfig(name="eval_test",
                                                       query_glob_path="s3://ai2-oe-data/jakep/pdfdata/openai_batch_data_v5_1_eval/*.jsonl",
                                                        response_glob_path="s3://ai2-oe-data/jakep/pdfdata/openai_batch_done_v5_1_eval/*.json")]),

            valid_data=DataConfig(seed=42,
                                  sources=[SourceConfig(name="eval_test",
                                                       query_glob_path="s3://ai2-oe-data/jakep/pdfdata/openai_batch_data_v5_1_eval/*.jsonl",
                                                        response_glob_path="s3://ai2-oe-data/jakep/pdfdata/openai_batch_done_v5_1_eval/*.json")])
        )
        train_dataset, valid_dataset = make_dataset(config, processor)    

        im_end_token_ids = processor.tokenizer("<|im_end|>\n", add_special_tokens=False)["input_ids"]


        #train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=False)
        for entry in train_dataset:
            print({x: (y.shape, y.dtype) for (x,y) in entry.items()})

            self.assertEqual(entry["input_ids"].dtype, np.int64)
            self.assertEqual(entry["attention_mask"].dtype, np.int64)
            self.assertEqual(entry["labels"].dtype, np.int64)
            self.assertEqual(entry["pixel_values"].dtype, np.float32)
            self.assertEqual(entry["image_grid_thw"].dtype, np.int64)
            
            # Extract input_ids and labels
            input_ids = entry["input_ids"]
            labels = entry["labels"]

            # 1. Verify that the last token is the end token
            # Ensure input_ids is long enough
            self.assertTrue(len(input_ids) >= len(im_end_token_ids), "Input IDs are shorter than the end token sequence.")

            # Compare the last tokens of input_ids with im_end_token_ids
            self.assertEqual(
                input_ids[-len(im_end_token_ids):].tolist(),
                im_end_token_ids,
                "The last tokens of input_ids do not match the end token sequence."
            )

            # 2. Ensure labels are masked correctly and match input_ids after the mask
            # Find where labels start being non-masked (-100 is the mask value)
            label_indices = np.where(labels != -100)[0]

            # There should be at least one label that is not masked
            self.assertTrue(len(label_indices) > 0, "No unmasked labels found in labels array.")

            first_label_index = label_indices[0]

            # Ensure the masked portion is at least 10 tokens long
            self.assertTrue(first_label_index >= 10, "Masked portion of labels is less than 10 tokens.")

            # Check that all values before first_label_index are -100
            self.assertTrue(
                np.all(labels[:first_label_index] == -100),
                "Labels before the first unmasked token are not all -100."
            )

            # Check that the unmasked labels match the corresponding input_ids
            self.assertTrue(
                np.array_equal(labels[first_label_index:], input_ids[first_label_index:]),
                "Unmasked labels do not match the corresponding input_ids."
            )

            # Optionally, verify that the last unmasked tokens in labels match the end token IDs
            unmasked_labels = labels[labels != -100]
            self.assertEqual(
                unmasked_labels[-len(im_end_token_ids):].tolist(),
                im_end_token_ids,
                "The last unmasked tokens in labels do not match the end token sequence."
            )

    def testTokenizationMatches(self):
        ds = build_batch_query_response_vision_dataset(
            query_glob_path="s3://ai2-oe-data/jakep/pdfdata/openai_batch_data_v5_1_eval/*.jsonl",
            response_glob_path="s3://ai2-oe-data/jakep/pdfdata/openai_batch_done_v5_1_eval/*.json",
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
                    {"type": "text", "text": build_finetuning_prompt(example["raw_page_text"])},
                ],
            },

            {
                "role": "assistant",
                "content": example["response"]
            }
        ]

        text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)

        # Decode image from base64
        main_image = Image.open(BytesIO(base64.b64decode(example["input_prompt_image_base64"])))

        width, height = main_image.size
        assert 1800 <= max(width, height) <= 2200, f"Image size {width}x{height} invalid"
        main_image = main_image.resize((width // 2, height // 2), Image.LANCZOS)


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

        print("Original tokenization")
        print(processor.tokenizer.decode(inference_inputs["input_ids"][0]))
        print("\n\n")

        print("Assembled tokenization")
        print(processor.tokenizer.decode(training_inputs["input_ids"]))
        print("\n\n")

        # Make sure that the token streams are the same
        self.assertEqual(processor.tokenizer.decode(inference_inputs["input_ids"][0]),
                         processor.tokenizer.decode(training_inputs["input_ids"]))

        # Make sure that the labels are masked with -100s properly
        # You only want the last assistant generation itself to be not -100, and thus contributing to the loss

        # Find the positions where labels are not -100
        non_masked_positions = training_inputs['labels'] != -100

        # Extract the tokens at those positions
        label_tokens = training_inputs['input_ids'][non_masked_positions]

        # Decode those tokens
        decoded_labels = processor.tokenizer.decode(label_tokens)
        assistant_response_with_end = example["response"] + "<|im_end|>\n"

        # Assert that the decoded labels match the assistant's response with <|im_end|>\n
        self.assertEqual(decoded_labels, assistant_response_with_end)