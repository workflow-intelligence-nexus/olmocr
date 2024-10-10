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

        #train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=False)
        for entry in train_dataset:
            print({x: y.shape for (x,y) in entry.items()})
            


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