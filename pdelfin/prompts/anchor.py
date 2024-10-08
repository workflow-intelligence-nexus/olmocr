# This file generates anchor text in a variety of different ways
# The goal here is to generate a bit of text which can be used to help prompt a VLM
# to better understand a document

# pdftotext
# pdfium
# pymupdf
# pypdf

# coherency score best of these three
import subprocess
import math
import ftfy
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
        options = {
            "pdftotext": _get_pdftotext(local_pdf_path, page),
            "pymupdf": _get_pymupdf(local_pdf_path, page),
            "pdfium": _get_pdfium(local_pdf_path, page),
            "pypdf_raw": _get_pypdf_raw(local_pdf_path, page)
        }

        scores = {label: get_document_coherency(text) for label, text in options.items()}

        best_option_label = max(scores, key=scores.get)
        best_option = options[best_option_label]

        print(f"topcoherency chosen: {best_option_label}")

        return best_option
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
    text_elements: List[TextElement]
    image_elements: List[ImageElement]

def _pdf_report(local_pdf_path: str, page: int) -> PageReport:
    reader = PdfReader(local_pdf_path)
    page = reader.pages[page - 1]
    resources = page.get("/Resources", {})
    xobjects = resources.get("/XObject", {})
    text_elements, image_elements = [], []
    

    def visitor_body(text, cm, tm, font_dict, font_size):
        txt2user = mult(tm, cm)
        text_elements.append(TextElement(text, txt2user[4], txt2user[5]))

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
                image_elements.append(ImageElement(xobject_name, BoundingBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))))

    page.extract_text(visitor_text=visitor_body, visitor_operand_before=visitor_op)

    return PageReport(
        mediabox=BoundingBox.from_rectangle(page.mediabox),
        text_elements=text_elements,
        image_elements=image_elements,
    )


def _merge_image_elements(images: List[ImageElement], tolerance: float=0.5) -> List[ImageElement]:
    n = len(images)
    parent = list(range(n))  # Initialize Union-Find parent pointers

    def find(i):
        # Find with path compression
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != i:
            parent_i = parent[i]
            parent[i] = root
            i = parent_i
        return root

    def union(i, j):
        # Union by attaching root of one tree to another
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    def bboxes_overlap(b1: BoundingBox, b2: BoundingBox, tolerance: float) -> bool:
        # Compute horizontal and vertical distances between boxes
        h_dist = max(0, max(b1.x0, b2.x0) - min(b1.x1, b2.x1))
        v_dist = max(0, max(b1.y0, b2.y0) - min(b1.y1, b2.y1))
        # Check if distances are within tolerance
        return h_dist <= tolerance and v_dist <= tolerance

    # Union overlapping images
    for i in range(n):
        for j in range(i + 1, n):
            if bboxes_overlap(images[i].bbox, images[j].bbox, tolerance):
                union(i, j)

    # Group images by their root parent
    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # Merge images in the same group
    merged_images = []
    for indices in groups.values():
        # Initialize merged bounding box
        merged_bbox = images[indices[0]].bbox
        merged_name = images[indices[0]].name

        for idx in indices[1:]:
            bbox = images[idx].bbox
            # Expand merged_bbox to include the current bbox
            merged_bbox = BoundingBox(
                x0=min(merged_bbox.x0, bbox.x0),
                y0=min(merged_bbox.y0, bbox.y0),
                x1=max(merged_bbox.x1, bbox.x1),
                y1=max(merged_bbox.y1, bbox.y1),
            )
            # Optionally, update the name
            merged_name += f"+{images[idx].name}"

        merged_images.append(ImageElement(name=merged_name, bbox=merged_bbox))

    # Return the merged images along with other elements
    return merged_images


def _linearize_pdf_report(report: PageReport) -> str:
    result = ""

    result += f"Page dimensions: {report.mediabox.x1:.1f}x{report.mediabox.y1:.1f}\n"

    #images = report.image_elements
    images = _merge_image_elements(report.image_elements)

    for index, element in enumerate(images):
        result += f"[Image {element.bbox.x0:.0f}x{element.bbox.y0:.0f} to {element.bbox.x1:.0f}x{element.bbox.y1:.0f}]"

    for index, element in enumerate(report.text_elements):
        if len(element.text.strip()) == 0:
            continue

        element_text = ftfy.fix_text(element.text)
        # Replace square brackets with something else not to throw off the syntax
        element_text = element_text.replace("[", "\[").replace("]", "\[")

        # Need to use ftfy to fix text, because occasionally there are invalid surrogate pairs and other UTF issues that cause
        # pyarrow to fail to load the json later
        result += f"[{element.x:.0f}x{element.y:.0f}]{element_text}"

    return result
