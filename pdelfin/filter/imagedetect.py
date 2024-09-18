import numpy as np
from pypdf import PdfReader
from pypdf.generic import ContentStream, NameObject, NumberObject


def process_content(content_stream, resources):
    total_image_area = 0
    graphics_state_stack = []
    current_matrix = np.eye(3)

    for operands, operator in content_stream.operations:
        if operator == b"q":  # Save graphics state
            graphics_state_stack.append(current_matrix.copy())
        elif operator == b"Q":  # Restore graphics state
            current_matrix = graphics_state_stack.pop()
        elif operator == b"cm":  # Concatenate matrix to CTM
            a, b, c, d, e, f = operands  # [a, b, c, d, e, f]
            cm_matrix = np.array([[a, b, 0], [c, d, 0], [e, f, 1]])
            current_matrix = np.matmul(current_matrix, cm_matrix)
        elif operator == b"Do":  # Paint external object
            xObjectName = operands[0]
            if "/XObject" in resources and xObjectName in resources["/XObject"]:
                xObject = resources["/XObject"][xObjectName]
                if xObject["/Subtype"] == "/Image":
                    width = xObject["/Width"]
                    height = xObject["/Height"]

                    # Calculate the area scaling factor using the absolute value of the determinant

                    image_area = float(width) * float(height) * np.linalg.det(current_matrix)
                    total_image_area += image_area
    return total_image_area


def pdf_page_image_area(reader: PdfReader, page_num: int) -> float:
    page = reader.pages[page_num - 1]

    page_width = float(page.mediabox.width)
    page_height = float(page.mediabox.height)
    page_area = page_width * page_height

    content = page.get_contents()
    if content is None:
        return float("nan")

    content_stream = ContentStream(content, reader)
    resources = page["/Resources"]

    image_area = process_content(content_stream, resources)

    return image_area / page_area
