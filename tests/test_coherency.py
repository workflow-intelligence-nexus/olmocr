import os

import unittest

from pdelfin.filter.coherency import get_document_coherency
from pdelfin.extract_text import get_document_text


class TestCoherencyScores(unittest.TestCase):
    def testBadOcr1(self):
        good_text = get_document_text(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "instructions_and_schematics.pdf"))
        bad_text = get_document_text(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "handwriting_bad_ocr.pdf"))

        print("Good", get_document_coherency(good_text))
        print("Bad", get_document_coherency(bad_text))