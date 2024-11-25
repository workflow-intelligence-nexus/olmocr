# The idea is that you have a Qwen2-VL-7B model located here:s3://ai2-oe-data/jakep/experiments/qwen2vl-pdf/v1/models/jakep/Qwen_Qwen2-VL-7B-Instruct-e4ecf8-01JAH8GMWHTJ376S2N7ETXRXH4/checkpoint-9500/bf16/"

# You need to load it in both hugging face transformers, and send page 1 of edgar.pdf to it from tests/gnarly_pdfs
# Compare that the temperature 0 sampled result is the same

import asyncio
import unittest
from unittest.mock import patch, AsyncMock
import os
import json
import tempfile
import math
import base64
import torch
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration
from pathlib import Path
from pdelfin.beakerpipeline import sglang_server_task, sglang_server_ready, build_page_query, SGLANG_SERVER_PORT, render_pdf_to_base64png, get_anchor_text, download_directory
from pdelfin.prompts import PageResponse
from httpx import AsyncClient
import torch.nn.functional as F
MODEL_FINETUNED_PATH = "s3://ai2-oe-data/jakep/experiments/qwen2vl-pdf/v1/models/jakep/Qwen_Qwen2-VL-7B-Instruct-e4ecf8-01JAH8GMWHTJ376S2N7ETXRXH4/checkpoint-9500/bf16/"

EDGAR_TEXT = (
    "Edgar, King of England\n\nEdgar (or Eadgar;[1] c. 944 – 8 July 975) was King of the English from 959 until his death in 975. "
    "He became king of all England on his brother's death. He was the younger son of King Edmund I and his first wife Ælfgifu. "
    "A detailed account of Edgar's reign is not possible, because only a few events were recorded by chroniclers and monastic writers "
    "were more interested in recording the activities of the leaders of the church.\n\nEdgar mainly followed the political policies of his predecessors, "
    "but there were major changes in the religious sphere. The English Benedictine Reform, which he strongly supported, became a dominant religious and social force.[2] "
    "It is seen by historians as a major achievement, and it was accompanied by a literary and artistic flowering, mainly associated with Æthelwold, Bishop of Winchester. "
    "Monasteries aggressively acquired estates from lay landowners with Edgar's assistance, leading to disorder when he died and former owners sought to recover their lost property, "
    "sometimes by force. Edgar's major administrative reform was the introduction of a standardised coinage in the early 970s to replace the previous decentralised system. "
    "He also issued legislative codes which mainly concentrated on improving procedures for enforcement of the law.\n\nEngland had suffered from Viking invasions for over a century "
    "when Edgar came to power, but there were none during his reign, which fell in a lull in attacks between the mid-950s and the early 980s.[3] After his death the throne was disputed "
    "between the supporters of his two surviving sons; the elder one, Edward the Martyr, was chosen with the support of Dunstan, the Archbishop of Canterbury. Three years later Edward was "
    "murdered and succeeded by his younger half-brother, Æthelred the Unready. Later chroniclers presented Edgar's reign as a golden age when England was free from external attacks and internal disorder, especially"
)

class TestSglangServer(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Mock arguments
        self.args = AsyncMock()
        self.args.workspace = "/tmp/test_workspace"
        self.args.model = [MODEL_FINETUNED_PATH]
        self.args.model_chat_template = "qwen2-vl"
        self.args.target_longest_image_dim = 1024
        self.args.target_anchor_text_len = 6000
        self.args.model_max_context = 8192

        # Create a temporary workspace directory
        os.makedirs(self.args.workspace, exist_ok=True)

        # Set up a semaphore for server tasks
        self.semaphore = asyncio.Semaphore(1)
        self.maxDiff = None

        # Start the sglang server
        self.my_server_task = asyncio.create_task(sglang_server_task(self.args, self.semaphore))
        
        # Wait for the server to become ready
        await sglang_server_ready()

    @patch("pdelfin.beakerpipeline.build_page_query", autospec=True)
    async def test_sglang_server_initialization_and_request(self, mock_build_page_query):
        # Mock the build_page_query function to set temperature to 0.0
        async def mocked_build_page_query(*args, **kwargs):
            query = await main.build_page_query(*args, **kwargs)
            query["temperature"] = 0.0  # Override temperature
            return query

        # Mock data paths
        self.test_pdf_path = Path(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "edgar.pdf"))

        # Send a single request to the sglang server for page 1
        async with AsyncClient(timeout=600) as session:
            query = await build_page_query(
                str(self.test_pdf_path),
                page=1,
                target_longest_image_dim=self.args.target_longest_image_dim,
                target_anchor_text_len=self.args.target_anchor_text_len,
            )
            COMPLETION_URL = f"http://localhost:{SGLANG_SERVER_PORT}/v1/chat/completions"
            response = await session.post(COMPLETION_URL, json=query)

        # Check the server response
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertIn("choices", response_data)
        self.assertGreater(len(response_data["choices"]), 0)

        model_response_json = json.loads(response_data["choices"][0]["message"]["content"])
        page_response = PageResponse(**model_response_json)

        print(page_response)

        self.assertEqual(page_response.natural_text, EDGAR_TEXT)


    async def asyncTearDown(self):
        # Shut down the server
        self.my_server_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await self.my_server_task

        # Cleanup temporary workspace
        if os.path.exists(self.args.workspace):
            for root, _, files in os.walk(self.args.workspace):
                for file in files:
                    os.unlink(os.path.join(root, file))
            os.rmdir(self.args.workspace)


class TestHuggingFaceModel(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Set up the Hugging Face model and tokenizer
        model_cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'pdelfin', 'model')
        download_directory([MODEL_FINETUNED_PATH], model_cache_dir)

        # Check the rope config and make sure it's got the proper key
        with open(os.path.join(model_cache_dir, "config.json"), "r") as cfin:
            config_data = json.load(cfin)

        if "rope_type" in config_data["rope_scaling"]:
            del config_data["rope_scaling"]["rope_type"]
            config_data["rope_scaling"]["type"] = "mrope"

            with open(os.path.join(model_cache_dir, "config.json"), "w") as cfout:
                json.dump(config_data, cfout)

        self.tokenizer = AutoTokenizer.from_pretrained(model_cache_dir, trust_remote_code=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_cache_dir, torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Path to the test PDF
        self.test_pdf_path = Path(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "edgar.pdf"))
        self.maxDiff = None

    async def test_hugging_face_generation(self):
        query = await build_page_query(
                str(self.test_pdf_path),
                page=1,
                target_longest_image_dim=1024,
                target_anchor_text_len=6000,
            )

       # Apply chat template to get the text
        text = self.processor.apply_chat_template(
            query["messages"], tokenize=False, add_generation_prompt=True
        )

        print(text)

        image_url = query["messages"][0]["content"][1]["image_url"]["url"]

        # Remove the "data:image/png;base64," prefix
        base64_image = image_url.split(",")[1]

        # Decode the base64 string into bytes
        image_data = base64.b64decode(base64_image)

        # Create a BytesIO object and load it into a PIL image
        main_image = Image.open(BytesIO(image_data))

        # Process inputs using processor
        inputs = self.processor(
            text=[text],
            images=[main_image],
            padding=True,
            return_tensors="pt",
        )

        inputs = {key: value.to(self.device) for (key, value) in inputs.items()}

        # Generate the output with temperature=0
        generation_output = self.model.generate(
            **inputs,
            temperature=0.0,
            max_new_tokens=1,
            max_length=8192,
            num_return_sequences=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        print(generation_output.scores)

        # Extract the generated token's log probabilities
        scores = generation_output.scores  # Tuple of length 1
        logits = scores[0]  # Tensor of shape (batch_size, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)  # Apply log softmax to get log probabilities

        # Get top 5 tokens and their log probabilities
        topk_log_probs, topk_indices = torch.topk(log_probs[0], k=5)
        topk_tokens = self.tokenizer.convert_ids_to_tokens(topk_indices.tolist())

        print("Top 5 tokens and their log probabilities:")
        for token, log_prob in zip(topk_tokens, topk_log_probs.tolist()):
            print(f"Token: {token}, Log Prob: {log_prob:.2f}, Prob {math.exp(log_prob):.2f}%")


        # # Decode the output
        # decoded_output = self.tokenizer.decode(generation_output[0], skip_special_tokens=True)

        # print(decoded_output)

        # # Convert the decoded output into the expected PageResponse structure
        # input_length = inputs["input_ids"].shape[1]

        # # Decode the output and extract only the new part
        # decoded_output = self.tokenizer.decode(generation_output[0], skip_special_tokens=True)
        # new_part = self.tokenizer.decode(generation_output[0][input_length:], skip_special_tokens=True)

        # print(new_part)

        # # Convert the new part into the expected PageResponse structure
        # generated_response = PageResponse(**json.loads(new_part))

        # # Assert the output matches the expected text
        # self.assertEqual(generated_response.natural_text, EDGAR_TEXT)

    async def asyncTearDown(self):
        # Clean up the model and tokenizer
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()