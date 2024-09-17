import os

import unittest

from pdelfin.filter.coherency import get_document_coherency
from pdelfin.extract_text import get_document_text, get_page_text


class TestCoherencyScores(unittest.TestCase):
    def testBadOcr1(self):
        good_text = get_document_text(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "instructions_and_schematics.pdf"))
        ocr1_text = get_document_text(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "handwriting_bad_ocr.pdf"))
        ocr2_text = get_document_text(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "some_ocr1.pdf"))

        print("Good", get_document_coherency(good_text))
        print("Bad1", get_document_coherency(ocr1_text))
        print("Bad2", get_document_coherency(ocr2_text))

    def testTwoColumnMisparse(self):
        pdftotext_text = get_page_text(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "pdftotext_two_column_issue.pdf"), page_num=2, pdf_engine="pdftotext")
        pymupdf_text = get_page_text(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "pdftotext_two_column_issue.pdf"), page_num=2, pdf_engine="pymupdf")

        print("pdftotext_text", get_document_coherency(pdftotext_text))
        print("pymupdf_text", get_document_coherency(pymupdf_text))

   