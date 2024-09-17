import unittest
import os

from pdelfin.filter import PdfFilter
from pdelfin.filter.imagedetect import pdf_page_image_area

from pypdf import PdfReader

class PdfFilterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.filter = PdfFilter()

    def testFormLaterPages(self):
        self.assertTrue(self.filter._is_form(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "form_on_later_pages.pdf")))

    
class ImageDetectionTest(unittest.TestCase):
    def testSlideshowMostlyImages(self):
        self.pdf = PdfReader(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "slideshow_mostly_images.pdf"))

        for page in range(self.pdf.get_num_pages()):
            print(page, pdf_page_image_area(self.pdf, page + 1))
    