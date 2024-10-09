import os
import unittest

from pypdf import PdfReader

from pdelfin.filter import PdfFilter


class PdfFilterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.filter = PdfFilter()

    def testFormLaterPages(self):
        self.assertTrue(
            self.filter._is_form(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "form_on_later_pages.pdf"))
        )

