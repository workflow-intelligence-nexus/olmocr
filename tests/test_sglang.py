# The idea is that you have a Qwen2-VL-7B model located here:s3://ai2-oe-data/jakep/experiments/qwen2vl-pdf/v1/models/jakep/Qwen_Qwen2-VL-7B-Instruct-e4ecf8-01JAH8GMWHTJ376S2N7ETXRXH4/checkpoint-9500/bf16/"

# You need to load it in both hugging face transformers, and send page 1 of edgar.pdf to it from tests/gnarly_pdfs
# Compare that the temperature 0 sampled result is the same

import asyncio
import unittest
from unittest.mock import patch, AsyncMock
import os
import json
import tempfile
from pathlib import Path
from pdelfin.beakerpipeline import sglang_server_task, sglang_server_ready, build_page_query, SGLANG_SERVER_PORT, render_pdf_to_base64png, get_anchor_text
from httpx import AsyncClient


class TestSglangServer(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Mock arguments
        self.args = AsyncMock()
        self.args.workspace = "/tmp/test_workspace"
        self.args.model = ["s3://ai2-oe-data/jakep/experiments/qwen2vl-pdf/v1/models/jakep/Qwen_Qwen2-VL-7B-Instruct-e4ecf8-01JAH8GMWHTJ376S2N7ETXRXH4/checkpoint-9500/bf16/"]
        self.args.model_chat_template = "qwen2-vl"
        self.args.target_longest_image_dim = 1024
        self.args.target_anchor_text_len = 6000
        self.args.model_max_context = 8192

        # Create a temporary workspace directory
        os.makedirs(self.args.workspace, exist_ok=True)

        # Set up a semaphore for server tasks
        self.semaphore = asyncio.Semaphore(1)

        # Mock data paths
        self.test_pdf_path = Path(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "edgar.pdf"))


    async def test_sglang_server_initialization_and_request(self):
        # Start the sglang server
        my_server_task = asyncio.create_task(sglang_server_task(self.args, self.semaphore))
        
        # Wait for the server to become ready
        await sglang_server_ready()

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

        # Shut down the server
        my_server_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await my_server_task

    async def asyncTearDown(self):
        # Cleanup temporary workspace
        if os.path.exists(self.args.workspace):
            for root, _, files in os.walk(self.args.workspace):
                for file in files:
                    os.unlink(os.path.join(root, file))
            os.rmdir(self.args.workspace)
