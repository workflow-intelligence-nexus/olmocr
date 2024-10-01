import unittest
import os
import json

from pypdf import PdfReader

class AnchorTest(unittest.TestCase):
    def testExtractText(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "pdftotext_two_column_issue.pdf")
        reader = PdfReader(local_pdf_path)
        page = reader.pages[1]

        def visitor_body(text, cm, tm, font_dict, font_size):
            print(repr(text))

        page.extract_text(visitor_text=visitor_body)

    def testAnchorBase(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "pdftotext_two_column_issue.pdf")

        from pdelfin.prompts._adv_anchor import extract_page
        reader = PdfReader(local_pdf_path)
        pypage = reader.pages[1]

        def visitor_body(text, cm, tm, font_dict, font_size):
            print(repr(text))

        extract_page(pypage, reader, visitor_text=visitor_body)

        # report = parse_pdf(local_pdf_path)
        # print(json.dumps(report, indent=1))

        # report = _pdf_report(local_pdf_path, 1)

        # print(json.dumps(report, indent=1))