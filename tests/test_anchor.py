import unittest
import os
import json

from pypdf import PdfReader

from pdelfin.prompts.anchor import _pdf_report, _linearize_pdf_report, get_anchor_text

class AnchorTest(unittest.TestCase):
    def testExtractText(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "some_ocr1.pdf")
        reader = PdfReader(local_pdf_path)
        page = reader.pages[0]

        def visitor_body(text, cm, tm, font_dict, font_size):
            print(repr(text), cm, tm, font_size)

        def visitor_op(op, args, cm, tm):
            #print(op, args, cm, tm)
            pass

        page.extract_text(visitor_text=visitor_body, visitor_operand_before=visitor_op)

    def testAnchorBase(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "pdftotext_two_column_issue.pdf")

        report = _pdf_report(local_pdf_path, 2)

        print(report)

        print(get_anchor_text(local_pdf_path, 2, pdf_engine="pdfreport"))

    def testAnchorImage(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "some_ocr1.pdf")

        report = _pdf_report(local_pdf_path, 1)

        print(report)

        print(get_anchor_text(local_pdf_path, 1, pdf_engine="pdfreport"))