import unittest
from unittest.mock import MagicMock, patch
import hashlib
import json
import os
import base64
from PIL import Image

# Adjust the import path to match where your code resides
from pdelfin.birrpipeline import build_dolma_doc, DatabaseManager, build_finetuning_prompt, build_page_query

class TestBuildDolmaDoc(unittest.TestCase):
    @patch('pdelfin.birrpipeline.DatabaseManager')
    @patch('pdelfin.birrpipeline.get_s3_bytes')
    def test_build_dolma_doc_with_multiple_page_entries(self, mock_get_s3_bytes, mock_DatabaseManager):
        # Mock DatabaseManager instance
        mock_db_instance = MagicMock()
        mock_DatabaseManager.return_value = mock_db_instance

        # Define the PDF record
        pdf_s3_path = 's3://bucket/pdf/test.pdf'
        pdf = DatabaseManager.PDFRecord(s3_path=pdf_s3_path, num_pages=1, status='pending')

        # Create multiple BatchInferenceRecord entries for page_num=1
        entry_a = DatabaseManager.BatchInferenceRecord(
            inference_s3_path='s3://bucket/inference/output1.jsonl',
            pdf_s3_path=pdf_s3_path,
            page_num=1,
            round=0,
            start_index=0,
            length=100,
            finish_reason='stop',
            error=None
        )

        entry_b = DatabaseManager.BatchInferenceRecord(
            inference_s3_path='s3://bucket/inference/output2.jsonl',
            pdf_s3_path=pdf_s3_path,
            page_num=1,
            round=0,
            start_index=0,
            length=100,
            finish_reason='stop',
            error=None
        )

        entry_c = DatabaseManager.BatchInferenceRecord(
            inference_s3_path='s3://bucket/inference/output3.jsonl',
            pdf_s3_path=pdf_s3_path,
            page_num=1,
            round=0,
            start_index=0,
            length=100,
            finish_reason='stop',
            error=None
        )

        entry_d = DatabaseManager.BatchInferenceRecord(
            inference_s3_path='s3://bucket/inference/output4.jsonl',
            pdf_s3_path=pdf_s3_path,
            page_num=1,
            round=0,
            start_index=0,
            length=100,
            finish_reason='stop',
            error=None
        )

        # Set up mock_db_instance.get_index_entries to return all entries
        mock_db_instance.get_index_entries.return_value = [entry_a, entry_b, entry_c, entry_d]

        # Define get_s3_bytes side effect function
        def get_s3_bytes_side_effect(s3_client, s3_path, start_index=None, end_index=None):
            if s3_path == 's3://bucket/inference/output1.jsonl':
                data = {
                    "custom_id": f"{pdf_s3_path}-1",
                    "outputs": [{"text": "{\"is_rotation_valid\": true, \"natural_text\": \"Short Text\"}"}],
                    "round": 0
                }
            elif s3_path == 's3://bucket/inference/output2.jsonl':
                data = {
                    "custom_id": f"{pdf_s3_path}-1",
                    "outputs": [{"text": "{\"is_rotation_valid\": false, \"natural_text\": \"Very Long Text Here that is longer\"}"}],
                    "round": 0
                }
            elif s3_path == 's3://bucket/inference/output3.jsonl':
                data = {
                    "custom_id": f"{pdf_s3_path}-1",
                    "outputs": [{"text": "{\"is_rotation_valid\": true, \"natural_text\": \"Medium Length Text\"}"}],
                    "round": 0
                }
            elif s3_path == 's3://bucket/inference/output4.jsonl':
                data = {
                    "custom_id": f"{pdf_s3_path}-1",
                    "outputs": [{"text": "{\"is_rotation_valid\": true, \"natural_text\": \"The Longest Correct Text\"}"}],
                    "round": 0
                }
            else:
                data = {}

            line = json.dumps(data) + '\n'
            content_bytes = line.encode('utf-8')
            return content_bytes

        mock_get_s3_bytes.side_effect = get_s3_bytes_side_effect

        # Call build_dolma_doc
        s3_workspace = 's3://bucket/workspace'
        dolma_doc = build_dolma_doc(s3_workspace, pdf)

        # Check that the resulting dolma_doc has the expected document_text
        expected_text = 'The Longest Correct Text\n'

        self.assertIsNotNone(dolma_doc)
        self.assertEqual(dolma_doc['text'], expected_text)

        # Additional assertions to ensure that the correct page was selected
        self.assertEqual(dolma_doc['metadata']['Source-File'], pdf_s3_path)
        self.assertEqual(dolma_doc['metadata']['pdf-total-pages'], 1)
        self.assertEqual(len(dolma_doc['attributes']['pdf_page_numbers']), 1)
        self.assertEqual(dolma_doc['attributes']['pdf_page_numbers'][0][2], 1)

        # Ensure that the document ID is correctly computed
        expected_id = hashlib.sha1(expected_text.encode()).hexdigest()
        self.assertEqual(dolma_doc['id'], expected_id)


class TestBuildPageQuery(unittest.TestCase):
    def testRotation(self):
        # First, generate and save the non-rotated image
        query = build_page_query(os.path.join(
            os.path.dirname(__file__),
            "gnarly_pdfs",
            "edgar.pdf"
        ), "edgar.pdf", 1, 1024, 6000, 0)

        # Extract the base64 image from the query
        image_content = query["chat_messages"][0]["content"][1]
        self.assertEqual(image_content["type"], "image_url")
        image_url = image_content["image_url"]["url"]

        # Extract base64 string from the data URL
        prefix = "data:image/png;base64,"
        self.assertTrue(image_url.startswith(prefix))
        image_base64 = image_url[len(prefix):]

        # Decode the base64 string
        image_data = base64.b64decode(image_base64)

        # Define the output file path for the non-rotated image
        output_image_path = os.path.join(os.path.dirname(__file__), "test_renders", "output_image.png")

        # Save the non-rotated image to a file
        with open(output_image_path, "wb") as image_file:
            image_file.write(image_data)

        # Now, generate and save the rotated image (90 degrees clockwise)
        query_rotated = build_page_query(os.path.join(
            os.path.dirname(__file__),
            "gnarly_pdfs",
            "edgar.pdf"
        ), "edgar.pdf", 1, 1024, 6000, 90)

        # Extract the base64 image from the rotated query
        image_content_rotated = query_rotated["chat_messages"][0]["content"][1]
        self.assertEqual(image_content_rotated["type"], "image_url")
        image_url_rotated = image_content_rotated["image_url"]["url"]

        # Extract base64 string from the data URL for the rotated image
        self.assertTrue(image_url_rotated.startswith(prefix))
        image_base64_rotated = image_url_rotated[len(prefix):]

        # Decode the base64 string for the rotated image
        image_data_rotated = base64.b64decode(image_base64_rotated)

        # Define the output file path for the rotated image
        output_image_rotated_path = os.path.join(os.path.dirname(__file__), "test_renders", "output_image_rotated90.png")

        # Save the rotated image to a file
        with open(output_image_rotated_path, "wb") as image_file_rotated:
            image_file_rotated.write(image_data_rotated)

        # Verification Step: Ensure the rotated image is 90 degrees clockwise rotated

        # Open both images using PIL
        with Image.open(output_image_path) as original_image:
            with Image.open(output_image_rotated_path) as rotated_image:

                # Compare pixel by pixel
                original_pixels = original_image.load()
                rotated_pixels = rotated_image.load()
                width, height = original_image.size

                self.assertEqual(width, rotated_image.size[1])
                self.assertEqual(height, rotated_image.size[0])

                for x in range(width):
                    for y in range(height):

                        self.assertEqual(
                            original_pixels[x, y], rotated_pixels[height - 1 - y, x],
                            f"Pixel mismatch at ({x}, {y})"
                        )

        print("Rotation verification passed: The rotated image is correctly rotated 90 degrees clockwise.")


# Run the test
if __name__ == '__main__':
    unittest.main()
