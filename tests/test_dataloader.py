import unittest
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial

from transformers import AutoProcessor

from pdelfin.train.dataloader import (
    build_batch_query_response_vision_dataset,
    extract_openai_batch_query,
    extract_openai_batch_response,
    load_jsonl_into_ds
)

from pdelfin.train.dataprep import batch_prepare_data_for_qwen2_training, prepare_data_for_qwen2_training


class TestBatchQueryResponseDataset(unittest.TestCase):
    def testLoadS3(self):
        ds = load_jsonl_into_ds("s3://ai2-oe-data/jakep/openai_batch_data_v2/*.jsonl", first_n_files=3)

        print(f"Loaded {len(ds)} entries")
        print(ds)
        print(ds["train"])

    def testCombinedQueryResponse(self):
        ds = build_batch_query_response_vision_dataset(
            query_glob_path="s3://ai2-oe-data/jakep/pdfdata/openai_batch_data_v5_1_train/*.jsonl",
            response_glob_path="s3://ai2-oe-data/jakep/pdfdata/openai_batch_done_v5_1_train/*.json",
        )

        print(ds)

        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        from pdelfin.train.dataprep import filter_by_max_seq_len
        ds = ds.filter(partial(filter_by_max_seq_len, processor=processor, max_prompt_len=1000))

        print(ds[0])

    def testLocalDS(self):
        ds = build_batch_query_response_vision_dataset(
            query_glob_path="/root/openai_batch_data_v5_1_train/*.jsonl",
            response_glob_path="/root/openai_batch_data_v5_1_train_done/*.json",
        )

        print(ds)

        ds.to_parquet("/root/trainds_parquet/bigds.parquet")

        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        from pdelfin.train.dataprep import filter_by_max_seq_len
        ds = ds.filter(partial(filter_by_max_seq_len, processor=processor, max_prompt_len=1000))

        print(ds[0])

    def testPlotSequenceLengthHistogram(self):
        import plotly.express as px  

        ds = build_batch_query_response_vision_dataset(
            query_glob_path="s3://ai2-oe-data/jakep/pdfdata/openai_batch_data_v5_1_eval/*.jsonl",
            response_glob_path="s3://ai2-oe-data/jakep/pdfdata/openai_batch_done_v5_1_eval/*.json",
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

        initial_len = len(ds)

        from pdelfin.train.dataprep import filter_by_max_seq_len
        ds = ds.filter(partial(filter_by_max_seq_len, processor=processor, max_prompt_len=2200, max_response_len=2200))

        formatted_dataset = ds.with_transform(partial(batch_prepare_data_for_qwen2_training, processor=processor))
        train_dataloader = DataLoader(formatted_dataset, batch_size=1, num_workers=30, shuffle=False)

        max_seen_len = 0
        steps = 0
        sequence_lengths = []  # List to store sequence lengths
        for entry in tqdm(train_dataloader):
            num_input_tokens = entry["input_ids"].shape[1]
            max_seen_len = max(max_seen_len, num_input_tokens)
            sequence_lengths.append(num_input_tokens)  # Collecting sequence lengths

            if steps % 100 == 0:
                print(f"Max input len {max_seen_len}")

            steps += 1

            # model.forward(**{k: v.to("cuda:0") for (k,v) in entry.items()})
        print(f"Max input len {max_seen_len}")
        print(f"Total elements before filtering: {initial_len}")
        print(f"Total elements after filtering: {steps}")

        # Plotting the histogram using Plotly
        fig = px.histogram(
            sequence_lengths,
            nbins=100,
            title="Distribution of Input Sequence Lengths",
            labels={'value': 'Sequence Length', 'count': 'Frequency'}
        )

        fig.write_image("sequence_lengths_histogram.png")

    def testExtractBatch(self):
        query_data = load_jsonl_into_ds("s3://ai2-oe-data/jakep/openai_batch_data_v2_mini/*.jsonl", first_n_files=3)
        query_data = query_data["train"]
        query_data = query_data.map(extract_openai_batch_query, remove_columns=query_data.column_names)

        print(query_data)
        print(query_data[0]["custom_id"], query_data[0]["input_prompt_text"])

    def testExtractResponse(self):
        response_data = load_jsonl_into_ds("s3://ai2-oe-data/jakep/openai_batch_done_v2/*.json", first_n_files=3)
        response_data = response_data["train"]

        response_data = response_data.map(extract_openai_batch_response, remove_columns=response_data.column_names)

        print(response_data)
        print(response_data[0])

    def testIterableDataset(self):
        dataset = build_batch_query_response_vision_dataset(
            query_glob_path="s3://ai2-oe-data/jakep/openai_batch_data_v2/*.jsonl",
            response_glob_path="s3://ai2-oe-data/jakep/openai_batch_done_v2/*.json",
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

        formatted_dataset = dataset.to_iterable_dataset(num_shards=64)
        formatted_dataset = formatted_dataset.map(partial(prepare_data_for_qwen2_training, processor=processor, add_batch_dim=True), remove_columns=formatted_dataset.column_names).filter(lambda x: x["input_ids"].shape[0] < 4500)

        for entry in formatted_dataset:
            print(entry)
            break