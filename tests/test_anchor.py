import unittest
import os
import json
import io
import glob

from pypdf import PdfReader

from pdelfin.prompts.anchor import _pdf_report, _linearize_pdf_report, get_anchor_text
from pdelfin.data.renderpdf import get_pdf_media_box_width_height

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

    def testSmallPage(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "small_page_size.pdf")

        report = _pdf_report(local_pdf_path, 1)

        print(report)

        print(get_anchor_text(local_pdf_path, 1, pdf_engine="pdfreport"))

    def testBadUTFSurrogatePairsGeneration(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "badlines.pdf")

        anchor_text = get_anchor_text(local_pdf_path, 4, pdf_engine="pdfreport")

        jsondata = json.dumps({
            "text": anchor_text
        })

        import pyarrow as pa
        import pyarrow.json as paj
        import pyarrow.compute as pc

        buffer = io.BytesIO(jsondata.encode('utf-8'))
        paj.read_json(buffer, read_options=paj.ReadOptions(use_threads=False, block_size=len(jsondata)))

    def testLargePromptHint1(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "large_prompt_hint1.pdf")

        anchor_text = get_anchor_text(local_pdf_path, 4, pdf_engine="pdfreport")

        print(anchor_text)
        print(len(anchor_text))
        self.assertLessEqual(len(anchor_text), 1000)

    def testLargePromptHint2(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "large_prompt_hint2.pdf")

        anchor_text = get_anchor_text(local_pdf_path, 2, pdf_engine="pdfreport")

        print(anchor_text)
        print(len(anchor_text))
        self.assertLessEqual(len(anchor_text), 4000)

    def testLargePromptHint3(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "large_prompt_hint3.pdf")

        anchor_text = get_anchor_text(local_pdf_path, 2, pdf_engine="pdfreport")

        print(anchor_text)
        print(len(anchor_text))
        self.assertLessEqual(len(anchor_text), 4000)

    def testNewsPaperPromptHint(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "newspaper.pdf")

        anchor_text = get_anchor_text(local_pdf_path, 1, pdf_engine="pdfreport")

        print(anchor_text)
        print(len(anchor_text))
        self.assertLessEqual(len(anchor_text), 4000)

    def testTobaccoPaperMissingParagraphs(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "tobacco_missed_tokens_pg1.pdf")

        anchor_text = get_anchor_text(local_pdf_path, 1, pdf_engine="pdfreport")

        print(anchor_text)
        print(len(anchor_text))
        self.assertLessEqual(len(anchor_text), 4000)

    def testAnchorOtherLengths(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "tobacco_missed_tokens_pg1.pdf")

        anchor_text = get_anchor_text(local_pdf_path, 1, pdf_engine="pdfreport", target_length=2000)

        print(anchor_text)
        print(len(anchor_text))
        self.assertLessEqual(len(anchor_text), 2000)

        anchor_text = get_anchor_text(local_pdf_path, 1, pdf_engine="pdfreport", target_length=6000)

        print(anchor_text)
        print(len(anchor_text))
        self.assertLessEqual(len(anchor_text), 6000)

    def testFailingAnchor(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "failing_anchor_pg4.pdf")

        anchor_text = get_anchor_text(local_pdf_path, 4, pdf_engine="pdfreport")

        print(anchor_text)
        print(len(anchor_text))
        self.assertLess(len(anchor_text), 4000)

    def testEmptyAnchor(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "tobacco_missed_tokens_pg1.pdf")

        anchor_text = get_anchor_text(local_pdf_path, 1, pdf_engine="pdfreport", target_length=0)

        self.assertEqual(anchor_text.strip(), "Page dimensions: 612.0x792.0")

    # TODO This one still fails
    def testExcessiveMapAnchor(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "map1.pdf")

        anchor_text = get_anchor_text(local_pdf_path, 1, pdf_engine="pdfreport", target_length=6000)

        print(anchor_text)
        print(len(anchor_text))
        self.assertLess(len(anchor_text), 4000)

class BuildSilverTest(unittest.TestCase):
    def testSmallPage(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "small_page_size.pdf")

        from pdelfin.data.buildsilver import build_page_query

        result = build_page_query(local_pdf_path, "s3://test.pdf", 1)

        from pdelfin.data.renderpdf import get_png_dimensions_from_base64

        base64data = result["body"]["messages"][0]["content"][1]["image_url"]["url"]

        if base64data.startswith("data:image/png;base64,"):
            base64data = base64data[22:]

        width, height = get_png_dimensions_from_base64(base64data)

        print(width, height)

        assert max(width, height) == 2048

class TestRenderPdf(unittest.TestCase):
    def testFastMediaBoxMatchesPyPdf(self):
        for file in glob.glob(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "*.pdf")):
            reader = PdfReader(file)
            print("checking", file)
            
            for page_num in range(1, len(reader.pages) + 1):
                w1, h1 = get_pdf_media_box_width_height(file, page_num)
                pypdfpage = reader.pages[page_num - 1]

                self.assertEqual(w1, pypdfpage.mediabox.width)
                self.assertEqual(h1, pypdfpage.mediabox.height)