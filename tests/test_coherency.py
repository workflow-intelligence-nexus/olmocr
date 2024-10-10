import html
import multiprocessing
import os
import time
import unittest


from pdelfin.filter.coherency import get_document_coherency

from pdelfin.prompts.anchor import get_anchor_text

class TestCoherencyScores(unittest.TestCase):
    def testBadOcr1(self):
        good_text = get_anchor_text(
            os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "instructions_and_schematics.pdf"), 1, pdf_engine="pdftotext"
        )
        ocr1_text = get_anchor_text(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "handwriting_bad_ocr.pdf"), 1, pdf_engine="pdftotext")
        ocr2_text = get_anchor_text(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "some_ocr1.pdf"), 1, pdf_engine="pdftotext")

        print("Good", get_document_coherency(good_text))
        print("Bad1", get_document_coherency(ocr1_text))
        print("Bad2", get_document_coherency(ocr2_text))

    def testHugeBookCoherencySpeed(self):
        base_text = get_anchor_text(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "ti89_guidebook.pdf"), 1, pdf_engine="pdftotext")
        print(f"ti89 book length: {len(base_text):,}")

        warmup = get_document_coherency(base_text[:1000])

        base_text = base_text[:40000]

        start = time.perf_counter()
        score = get_document_coherency(base_text)
        end = time.perf_counter()

        char_per_sec = len(base_text) / (end - start)
        char_per_sec = char_per_sec / multiprocessing.cpu_count()

        print(f"ti89 book score {score:.2f}")
        print(f"{char_per_sec:.2f} chars per second per core")

    def testTwoColumnMisparse(self):
        pdftotext_text = get_anchor_text(
            os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "pdftotext_two_column_issue.pdf"),
            page=2,
            pdf_engine="pdftotext",
        )
        pymupdf_text = get_anchor_text(
            os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "pdftotext_two_column_issue.pdf"),
            page=2,
            pdf_engine="pymupdf",
        )
        pdfium_text = get_anchor_text(
            os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "pdftotext_two_column_issue.pdf"),
            page=2,
            pdf_engine="pdfium",
        )

        # pdftotext_text = get_document_text(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "pdftotext_two_column_issue.pdf"), pdf_engine="pdftotext")
        # pymupdf_text = get_document_text(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "pdftotext_two_column_issue.pdf"), pdf_engine="pymupdf")

        print("pdftotext_text", pdftotext_score := get_document_coherency(pdftotext_text))
        print("pymupdf_text", pymupdf_score := get_document_coherency(pymupdf_text))
        print("pdfium_text", pdfium_score := get_document_coherency(pdfium_text))

        self.assertLess(pdftotext_score, pymupdf_score)
        self.assertLess(pdfium_score, pymupdf_score)

        anchor_text = get_anchor_text(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "pdftotext_two_column_issue.pdf"), 2, pdf_engine="topcoherency")

        self.assertEqual(anchor_text, pymupdf_text)
