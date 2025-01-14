import argparse
import boto3
import dataclasses
import random
import re

from tqdm import tqdm
from itertools import combinations
from pdelfin.s3_utils import parse_s3_path, expand_s3_glob, get_s3_bytes
from dolma_refine.evaluate.metrics import DocumentEditSimilarity
from dolma_refine.evaluate.segmenters import SpacySegmenter
from dolma_refine.evaluate.aligners import HirschbergAligner

from pdelfin.eval.evalhtml import create_review_html

s3_client = boto3.client('s3')

@dataclasses.dataclass
class Comparison:
    pdf_path: str

    comparison_a_path: str
    comparison_b_path: str

    comparison_a_str: str
    comparison_b_str: str

    alignment: float

    @property
    def comparison_a_method(self):
        return re.search(r'page[0-9]+_(\w+)\.md$', self.comparison_a_path).group(1)

    @property
    def comparison_b_method(self):
        return re.search(r'page[0-9]+_(\w+)\.md$', self.comparison_b_path).group(1)

def build_review_page(args, comparisons):
    page_data = []

    for comp in comparisons:
       page_data.append({
                "s3_path": comp.pdf_path,
                "page": 1,
                "entry_key": comp.pdf_path + "-" + comp.comparison_a_method + "-" + comp.comparison_b_method,
                "gold_text": comp.comparison_a_str,
                "gold_metadata": comp.comparison_a_method,
                "eval_text": comp.comparison_b_str,
                "eval_metadata": comp.comparison_b_method,
                "alignment": comp.alignment
            })
       
    create_review_html(page_data, args.name + ".html")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates comparison voting pages between different pairs of parses for a PDF."
    )
    parser.add_argument(
        '--name',
        default="review_page",
        help="What name to give to this evaluation/comparison"
    )
    parser.add_argument(
        '--review_size',
        default=20,
        type=int,
        help="Number of entries to show on the generated review page",
    )
    parser.add_argument(
        '--comparisons',
        default=["pdelf", "gotocr", "gotocr_format"],
        help="Different variants to compare against"
    )
    parser.add_argument(
        's3_path',
        type=str,
        help='Path to the folder where you keep your data files, expecting to see *.md files in there along with *.png and *.pdf'
    )

    args = parser.parse_args()

    segmenter = SpacySegmenter("spacy")
    aligner = HirschbergAligner(match_score=1,
                                mismatch_score=-1,
                                indel_score=-1)
    comparer = DocumentEditSimilarity(segmenter=segmenter, aligner=aligner)

    all_comps = []
    all_pdfs = set(expand_s3_glob(s3_client, args.s3_path + "/*.pdf"))
    all_mds = set(expand_s3_glob(s3_client, args.s3_path + "/*.md"))

    for pdf_path in tqdm(all_pdfs):
        pdf_comps = []
        for comp in args.comparisons:
            comp_path = pdf_path.replace(".pdf", f"_{comp}.md")
            if comp_path in all_mds:
                pdf_comps.append(comp_path)
        
        for (compa, compb) in combinations(pdf_comps, 2):
            if random.choice([True, False]):
                compa, compb = compb, compa

            text_a = get_s3_bytes(s3_client, compa).decode("utf-8")
            text_b = get_s3_bytes(s3_client, compb).decode("utf-8")

            all_comps.append(
                Comparison(pdf_path=pdf_path,
                comparison_a_path=compa,
                comparison_b_path=compb,
                comparison_a_str=text_a,
                comparison_b_str=text_b,
                alignment=comparer.compute(text_a, text_b)
                )
            )
    
        # DEBUG CODE, remove
        if len(all_comps) > 10:
            break
            
    all_comps.sort(key=lambda c: c.alignment)
    result = build_review_page(args, all_comps[0:args.review_size])