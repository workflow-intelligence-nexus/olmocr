# This file generates anchor text in a variety of different ways
# The goal here is to generate a bit of text which can be used to help prompt a VLM
# to better understand a document

# pdftotext
# pdfium
# pymupdf
# pypdf

# coherency score best of these three
import subprocess
from typing import Literal

from pypdf import PdfReader
import pypdfium2 as pdfium
import pymupdf

from pdelfin.filter.coherency import get_document_coherency


def get_anchor_text(local_pdf_path: str, page: int, pdf_engine: Literal["pdftotext", "pdfium", "pymupdf", "pypdf", "topcoherency"]) -> str:
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

        # return option with the lowest score
        return options[scores.index(min(scores))]


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

    return textpage

