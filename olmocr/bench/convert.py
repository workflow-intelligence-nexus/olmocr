import argparse
import os
import glob

from tqdm import tqdm

# Import all of the runners
from olmocr.bench.runners.run_gotocr import run_gotocr
from olmocr.bench.runners.run_marker import run_marker

# Goes through each pdf in the data folder, and converts them with each provided method

if __name__ == "__main__":
    data_directory = os.path.join(os.path.dirname(__file__), "sample_data")
    pdf_directory = os.path.join(data_directory, "pdfs")
    
    config = {
        "marker": {
            "method": run_marker
        },

        "got_ocr": {
            "method": run_gotocr,
            "temperature": 0.0,
        },
    }


    for candidate in config.keys():
        print(f"Starting conversion using {candidate}")
        os.makedirs(os.path.join(data_directory, candidate), exist_ok=True)

        for pdf_path in tqdm(glob.glob(os.path.join(pdf_directory, "*.pdf")), desc=candidate):
            markdown = config[candidate]["method"](pdf_path, page_num=1)

            with open(os.path.join(data_directory, candidate, os.path.basename(pdf_path).replace(".pdf", ".md")), "w") as out_f:
                out_f.write(markdown)

            