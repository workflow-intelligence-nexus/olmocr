import csv
import datetime
import hashlib
import json
import logging
import math
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from io import StringIO

import requests
from lingua import Language, LanguageDetectorBuilder
from pypdf import PdfReader
from pypdf.errors import DependencyError, PyPdfError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PdfFilter:
    def __init__(self):
        super().__init__()
        self.language_detector = (
            LanguageDetectorBuilder.from_all_languages()
            .with_preloaded_language_models()
            .build()
        )
        self.ngram_log_probs = self._build_ngram_log_probs()

    # Used for comparing frequency of words to eliminate bad documents
    def _build_ngram_log_probs(self):
        NGRAM_DATASET_LINK = "https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/lucas/google-1T-unigram/unigram_freq.csv"

        ngrams = {}

        # Download the dataset
        response = requests.get(NGRAM_DATASET_LINK)
        if response.status_code != 200:
            raise Exception(
                f"Failed to download data, status code: {response.status_code}"
            )

        # Read the CSV content
        csv_content = StringIO(response.text)
        reader = csv.DictReader(csv_content)

        # Build the frequency dictionary
        total_count = 0

        for row in reader:
            word = row["word"]
            count = int(row["count"])
            total_count += count
            ngrams[word] = count

        # Convert to log probs
        return {word: math.log(count / total_count) for word, count in ngrams.items()}

    def _is_form(self, local_pdf_path: str) -> bool:
        # Remove PDFs which are forms
        try:
            pdf_reader = PdfReader(local_pdf_path)
            if pdf_reader.get_form_text_fields():
                return True
        except PyPdfError as pex:
            logger.exception(pex)
            logger.warning("Invalid PDF, filtering out")
            return False
        except DependencyError as dex:
            logger.warning(f"PDF requires external dependency {dex}, filtering out")
            return False
        except Exception as ex:
            logger.exception(ex)
            logger.warning(f"Internal error reading PDF, filtering out")
            return False

        # TODO: If distribution of _ characters is very high, it's probably a form

    def _is_download_spam(self, base_text: str, threshold: float = 0.004) -> bool:
        seo_words = {
            "download",
            "pdf",
            "epub",
            "mobi",
            "free",
            "ebook",
            "file",
            "save",
            "casino",
        }
        seo_word_probs = {word: self.ngram_log_probs[word] for word in seo_words}

        base_text = base_text.strip().lower()
        clean_text = re.sub(r"\W+", " ", base_text)

        word_counts = Counter(clean_text.split())
        total_words = len(clean_text.split())

        seo_score = sum(word_counts[word] for word in seo_words if word in word_counts)

        return seo_score / total_words > threshold

    # Returns True if there is something wrong with this PDF
    def filter_out_pdf(self, local_pdf_path: str) -> bool:
        # Basic metadata-level filtering
        if self._is_form(local_pdf_path):
            return False

        # Read the first five pages of text for language calculation
        pdftotext_result = subprocess.run(
            ["pdftotext", "-f", "1", "-l", "5", local_pdf_path, "-"],
            timeout=60,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if pdftotext_result.returncode != 0:
            logger.warn(
                f"pdftotext returned {pdftotext_result.returncode} on {local_pdf_path}"
            )
            return False

        base_text = pdftotext_result.stdout.decode("utf-8")

        # Other filter ideas:
        #  - Remove patents, they tend to be ocred, multicolumn, and should come in through a cleaner dataset
        #  - Detect things with too many figures
        #  - Detect too many pages with no input
        #  - Off distribution in terms of words per page, etc
        if len(base_text) < 100 or len(base_text.split()) < 50:
            logger.warn("PDF is too short, skipping")
            return False

        language = self.language_detector.detect_language_of(base_text)

        if language != Language.ENGLISH:
            logger.info(
                f"Filtering out {local_pdf_path} because language was {language}"
            )
            return True

        if self._is_download_spam(base_text):
            logger.info(f"Filtering out {local_pdf_path} because of SEO/download spam")
            return True

        return False
