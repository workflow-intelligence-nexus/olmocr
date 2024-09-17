import subprocess
import pymupdf

from typing import Literal


def get_page_text(local_pdf_path: str, page_num: int, pdf_engine: Literal["pdftotext", "pymupdf", "pdfium"]="pdftotext") -> str:
    if pdf_engine == "pdftotext":
        pdftotext_result = subprocess.run(
            [
                "pdftotext",
                "-f",
                str(page_num),
                "-l",
                str(page_num),
                local_pdf_path,
                "-",
            ],
            timeout=60,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert pdftotext_result.returncode == 0

        return pdftotext_result.stdout.decode("utf-8")
    elif pdf_engine == "pymupdf":
        pm_doc = pymupdf.open(local_pdf_path)
        return pm_doc[page_num - 1].get_text()
    else:
        raise NotImplementedError()


def get_document_text(local_pdf_path: str, pdf_engine: Literal["pdftotext", "pymupdf", "pdfium"]="pdftotext") -> str:
    if pdf_engine == "pdftotext":
        pdftotext_result = subprocess.run(
            [
                "pdftotext",
                local_pdf_path,
                "-",
            ],
            timeout=60,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert pdftotext_result.returncode == 0

        return pdftotext_result.stdout.decode("utf-8")
    elif pdf_engine == "pymupdf":
        pm_doc = pymupdf.open(local_pdf_path)
        result = ""

        for page in pm_doc:
            result += page.get_text()
            result += "\n"

        return result
    else:
        raise NotImplementedError()