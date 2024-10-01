# This file generates anchor text in a variety of different ways
# The goal here is to generate a bit of text which can be used to help prompt a VLM
# to better understand a document

# pdftotext
# pdfium
# pymupdf
# pypdf

# coherency score best of these three
import subprocess
import sys
import json
from dataclasses import dataclass
from typing import Literal, List

import pypdfium2 as pdfium
import pymupdf

from pdelfin.filter.coherency import get_document_coherency

from pypdf import PdfReader
from pypdf.generic import RectangleObject
from pdelfin.prompts._adv_anchor import mult


def get_anchor_text(local_pdf_path: str, page: int, pdf_engine: Literal["pdftotext", "pdfium", "pymupdf", "pypdf", "topcoherency", "pdfreport"]) -> str:
    assert page > 0, "Pages are 1-indexed in pdf-land"

    if pdf_engine == "pdftotext":
        return _get_pdftotext(local_pdf_path, page)
    elif pdf_engine == "pdfium":
        return _get_pdfium(local_pdf_path, page)
    elif pdf_engine == "pypdf":
        return _get_pypdf_raw(local_pdf_path, page)
    elif pdf_engine == "pymupdf":
        return _get_pymupdf(local_pdf_path, page)
    elif pdf_engine == "topcoherency":
        options = [
            _get_pdftotext(local_pdf_path, page),
            _get_pymupdf(local_pdf_path, page),
            _get_pdfium(local_pdf_path, page),
            _get_pypdf_raw(local_pdf_path, page)
        ]

        scores = [get_document_coherency(text) for text in options]

        # return option with the best (highest) score (higher is more likley, as these are logprobs)
        return options[scores.index(max(scores))]
    elif pdf_engine == "pdfreport":
        return _linearize_pdf_report(_pdf_report(local_pdf_path, page))
    else:
        raise NotImplementedError("Unknown engine")


def _get_pdftotext(local_pdf_path: str, page: int) -> str:
    pdftotext_result = subprocess.run(
        ["pdftotext", "-f", str(page), "-l", str(page), local_pdf_path, "-"],
        timeout=60,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert pdftotext_result.returncode == 0
    return pdftotext_result.stdout.decode("utf-8")

def _get_pymupdf(local_pdf_path: str, page: int) -> str:
    pm_doc = pymupdf.open(local_pdf_path)
    return pm_doc[page - 1].get_text()

def _get_pypdf_raw(local_pdf_path: str, page: int) -> str:
    reader = PdfReader(local_pdf_path)
    pypage = reader.pages[page - 1]

    return pypage.extract_text()

def _get_pdfium(local_pdf_path: str, page: int) -> str:
    pdf = pdfium.PdfDocument(local_pdf_path)
    textpage = pdf[page - 1].get_textpage()
    return textpage.get_text_bounded()

def _transform_point(x, y, m):
    x_new = m[0]*x + m[2]*y + m[4]
    y_new = m[1]*x + m[3]*y + m[5]
    return x_new, y_new

@dataclass
class Element:
    pass

@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @staticmethod
    def from_rectangle(rect: RectangleObject) -> "BoundingBox":
        return BoundingBox(rect[0], rect[1], rect[2], rect[3])


@dataclass
class TextElement(Element):
    text: str
    x: float
    y: float

@dataclass
class ImageElement(Element):
    name: str
    bbox: BoundingBox

@dataclass
class PageReport:
    mediabox: BoundingBox
    elements: List[Element]

def _pdf_report(local_pdf_path: str, page: int) -> PageReport:
    reader = PdfReader(local_pdf_path)
    page = reader.pages[page - 1]
    resources = page.get("/Resources", {})
    xobjects = resources.get("/XObject", {})
    elements = []

    def visitor_body(text, cm, tm, font_dict, font_size):
        txt2user = mult(tm, cm)
        elements.append(TextElement(text, txt2user[4], txt2user[5]))

    def visitor_op(op, args, cm, tm):
        if op == b"Do":
            xobject_name = args[0]
            xobject = xobjects.get(xobject_name)
            if xobject and xobject["/Subtype"] == "/Image":
                # Compute image bbox
                # The image is placed according to the CTM
                width = xobject.get("/Width")
                height = xobject.get("/Height")
                x0, y0 = _transform_point(0, 0, cm)
                x1, y1 = _transform_point(1, 1, cm)
                elements.append(ImageElement(xobject_name, BoundingBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))))

    page.extract_text(visitor_text=visitor_body, visitor_operand_before=visitor_op)

    return PageReport(
        mediabox=BoundingBox.from_rectangle(page.mediabox),
        elements=elements,
    )


def _linearize_pdf_report(report: PageReport) -> str:
    result = ""

    result += f"Page dimensions: {report.mediabox.x1:.1f}x{report.mediabox.y1:.1f}\n"
    
    for index, element in enumerate(report.elements):
        if isinstance(element, ImageElement):
            result += f"[Image {element.bbox.x0:.0f}x{element.bbox.y0:.0f} to {element.bbox.x1:.0f}x{element.bbox.y1:.0f}]"
        if isinstance(element, TextElement):
            if len(element.text.strip()) == 0:
                continue

            result += f"[{element.x:.0f}x{element.y:.0f}]{element.text}"

    return result
