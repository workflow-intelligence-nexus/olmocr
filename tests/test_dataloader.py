import unittest

from pdelfin.train.dataloader import load_jsonl_from_s3, build_batch_query_response_vision_dataset
from pdelfin.train.dataloader import extract_openai_batch_query, extract_openai_batch_response

class TestBatchQueryResponseDataset(unittest.TestCase):
    def testLoadS3(self):
        ds = load_jsonl_from_s3("s3://ai2-oe-data/jakep/openai_batch_data_v2/*.jsonl", first_n_files=3)

        print(f"Loaded {len(ds)} entries")
        print(ds)
        print(ds["train"])
    
    def testCombinedQueryResponse(self):
        ds = build_batch_query_response_vision_dataset(query_glob_path="s3://ai2-oe-data/jakep/openai_batch_data_v2/*.jsonl",
                                                       response_glob_path="s3://ai2-oe-data/jakep/openai_batch_done_v2/*.json")

        print(ds)

    def testExtractBatch(self):
        query_data = load_jsonl_from_s3("s3://ai2-oe-data/jakep/openai_batch_data_v2/*.jsonl", first_n_files=3)
        query_data = query_data["train"]
        query_data = query_data.map(extract_openai_batch_query, remove_columns=query_data.column_names)

        print(query_data)
        print(query_data[0]["custom_id"], query_data[0]["input_prompt_text"])

    def testExtractResponse(self):
        response_data = load_jsonl_from_s3("s3://ai2-oe-data/jakep/openai_batch_done_v2/*.json", first_n_files=3)
        response_data = response_data["train"]

        response_data = response_data.map(extract_openai_batch_response, remove_columns=response_data.column_names)

        print(response_data)
        print(response_data[0])
