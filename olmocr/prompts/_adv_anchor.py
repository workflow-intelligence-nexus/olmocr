import math
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

from pypdf._cmap import build_char_map, unknown_char_map
from pypdf.constants import PageAttributes as PG
from pypdf.generic import (
    ContentStream,
    DictionaryObject,
    FloatObject,
    NameObject,
    NumberObject,
    TextStringObject,
    encode_pdfdocencoding,
)

CUSTOM_RTL_MIN: int = -1
CUSTOM_RTL_MAX: int = -1
CUSTOM_RTL_SPECIAL_CHARS: List[int] = []
LAYOUT_NEW_BT_GROUP_SPACE_WIDTHS: int = 5


class OrientationNotFoundError(Exception):
    pass


def set_custom_rtl(
    _min: Union[str, int, None] = None,
    _max: Union[str, int, None] = None,
    specials: Union[str, List[int], None] = None,
) -> Tuple[int, int, List[int]]:
    """
    Change the Right-To-Left and special characters custom parameters.

    Args:
        _min: The new minimum value for the range of custom characters that
            will be written right to left.
            If set to ``None``, the value will not be changed.
            If set to an integer or string, it will be converted to its ASCII code.
            The default value is -1, which sets no additional range to be converted.
        _max: The new maximum value for the range of custom characters that will
            be written right to left.
            If set to ``None``, the value will not be changed.
            If set to an integer or string, it will be converted to its ASCII code.
            The default value is -1, which sets no additional range to be converted.
        specials: The new list of special characters to be inserted in the
            current insertion order.
            If set to ``None``, the current value will not be changed.
            If set to a string, it will be converted to a list of ASCII codes.
            The default value is an empty list.

    Returns:
        A tuple containing the new values for ``CUSTOM_RTL_MIN``,
        ``CUSTOM_RTL_MAX``, and ``CUSTOM_RTL_SPECIAL_CHARS``.
    """
    global CUSTOM_RTL_MIN, CUSTOM_RTL_MAX, CUSTOM_RTL_SPECIAL_CHARS
    if isinstance(_min, int):
        CUSTOM_RTL_MIN = _min
    elif isinstance(_min, str):
        CUSTOM_RTL_MIN = ord(_min)
    if isinstance(_max, int):
        CUSTOM_RTL_MAX = _max
    elif isinstance(_max, str):
        CUSTOM_RTL_MAX = ord(_max)
    if isinstance(specials, str):
        CUSTOM_RTL_SPECIAL_CHARS = [ord(x) for x in specials]
    elif isinstance(specials, list):
        CUSTOM_RTL_SPECIAL_CHARS = specials
    return CUSTOM_RTL_MIN, CUSTOM_RTL_MAX, CUSTOM_RTL_SPECIAL_CHARS


def mult(m: List[float], n: List[float]) -> List[float]:
    return [
        m[0] * n[0] + m[1] * n[2],
        m[0] * n[1] + m[1] * n[3],
        m[2] * n[0] + m[3] * n[2],
        m[2] * n[1] + m[3] * n[3],
        m[4] * n[0] + m[5] * n[2] + n[4],
        m[4] * n[1] + m[5] * n[3] + n[5],
    ]


def orient(m: List[float]) -> int:
    if m[3] > 1e-6:
        return 0
    elif m[3] < -1e-6:
        return 180
    elif m[1] > 0:
        return 90
    else:
        return 270


def crlf_space_check(
    text: str,
    cmtm_prev: Tuple[List[float], List[float]],
    cmtm_matrix: Tuple[List[float], List[float]],
    memo_cmtm: Tuple[List[float], List[float]],
    cmap: Tuple[Union[str, Dict[int, str]], Dict[str, str], str, Optional[DictionaryObject]],
    orientations: Tuple[int, ...],
    output: str,
    font_size: float,
    visitor_text: Optional[Callable[[Any, Any, Any, Any, Any], None]],
    spacewidth: float,
) -> Tuple[str, str, List[float], List[float]]:
    cm_prev = cmtm_prev[0]
    tm_prev = cmtm_prev[1]
    cm_matrix = cmtm_matrix[0]
    tm_matrix = cmtm_matrix[1]
    memo_cm = memo_cmtm[0]
    memo_tm = memo_cmtm[1]

    m_prev = mult(tm_prev, cm_prev)
    m = mult(tm_matrix, cm_matrix)
    orientation = orient(m)
    delta_x = m[4] - m_prev[4]
    delta_y = m[5] - m_prev[5]
    k = math.sqrt(abs(m[0] * m[3]) + abs(m[1] * m[2]))
    f = font_size * k
    cm_prev = m
    if orientation not in orientations:
        raise OrientationNotFoundError
    try:
        if orientation == 0:
            if delta_y < -0.8 * f:
                if (output + text)[-1] != "\n":
                    output += text + "\n"
                    if visitor_text is not None:
                        visitor_text(
                            text + "\n",
                            memo_cm,
                            memo_tm,
                            cmap[3],
                            font_size,
                        )
                    text = ""
            elif abs(delta_y) < f * 0.3 and abs(delta_x) > spacewidth * f * 15 and (output + text)[-1] != " ":
                text += " "
        elif orientation == 180:
            if delta_y > 0.8 * f:
                if (output + text)[-1] != "\n":
                    output += text + "\n"
                    if visitor_text is not None:
                        visitor_text(
                            text + "\n",
                            memo_cm,
                            memo_tm,
                            cmap[3],
                            font_size,
                        )
                    text = ""
            elif abs(delta_y) < f * 0.3 and abs(delta_x) > spacewidth * f * 15 and (output + text)[-1] != " ":
                text += " "
        elif orientation == 90:
            if delta_x > 0.8 * f:
                if (output + text)[-1] != "\n":
                    output += text + "\n"
                    if visitor_text is not None:
                        visitor_text(
                            text + "\n",
                            memo_cm,
                            memo_tm,
                            cmap[3],
                            font_size,
                        )
                    text = ""
            elif abs(delta_x) < f * 0.3 and abs(delta_y) > spacewidth * f * 15 and (output + text)[-1] != " ":
                text += " "
        elif orientation == 270:
            if delta_x < -0.8 * f:
                if (output + text)[-1] != "\n":
                    output += text + "\n"
                    if visitor_text is not None:
                        visitor_text(
                            text + "\n",
                            memo_cm,
                            memo_tm,
                            cmap[3],
                            font_size,
                        )
                    text = ""
            elif abs(delta_x) < f * 0.3 and abs(delta_y) > spacewidth * f * 15 and (output + text)[-1] != " ":
                text += " "
    except Exception:
        pass
    tm_prev = tm_matrix.copy()
    cm_prev = cm_matrix.copy()
    return text, output, cm_prev, tm_prev


def handle_tj(
    text: str,
    operands: List[Union[str, TextStringObject]],
    cm_matrix: List[float],
    tm_matrix: List[float],
    cmap: Tuple[Union[str, Dict[int, str]], Dict[str, str], str, Optional[DictionaryObject]],
    orientations: Tuple[int, ...],
    output: str,
    font_size: float,
    rtl_dir: bool,
    visitor_text: Optional[Callable[[Any, Any, Any, Any, Any], None]],
) -> Tuple[str, bool]:
    m = mult(tm_matrix, cm_matrix)
    orientation = orient(m)
    if orientation in orientations and len(operands) > 0:
        if isinstance(operands[0], str):
            text += operands[0]
        else:
            t: str = ""
            tt: bytes = encode_pdfdocencoding(operands[0]) if isinstance(operands[0], str) else operands[0]
            if isinstance(cmap[0], str):
                try:
                    t = tt.decode(cmap[0], "surrogatepass")  # apply str encoding
                except Exception:
                    # the data does not match the expectation,
                    # we use the alternative ;
                    # text extraction may not be good
                    t = tt.decode(
                        "utf-16-be" if cmap[0] == "charmap" else "charmap",
                        "surrogatepass",
                    )  # apply str encoding
            else:  # apply dict encoding
                t = "".join([cmap[0][x] if x in cmap[0] else bytes((x,)).decode() for x in tt])
            # "\u0590 - \u08FF \uFB50 - \uFDFF"
            for x in [cmap[1][x] if x in cmap[1] else x for x in t]:
                # x can be a sequence of bytes ; ex: habibi.pdf
                if len(x) == 1:
                    xx = ord(x)
                else:
                    xx = 1
                # fmt: off
                if (
                    # cases where the current inserting order is kept
                    (xx <= 0x2F)                        # punctuations but...
                    or 0x3A <= xx <= 0x40               # numbers (x30-39)
                    or 0x2000 <= xx <= 0x206F           # upper punctuations..
                    or 0x20A0 <= xx <= 0x21FF           # but (numbers) indices/exponents
                    or xx in CUSTOM_RTL_SPECIAL_CHARS   # customized....
                ):
                    text = x + text if rtl_dir else text + x
                elif (  # right-to-left characters set
                    0x0590 <= xx <= 0x08FF
                    or 0xFB1D <= xx <= 0xFDFF
                    or 0xFE70 <= xx <= 0xFEFF
                    or CUSTOM_RTL_MIN <= xx <= CUSTOM_RTL_MAX
                ):
                    if not rtl_dir:
                        rtl_dir = True
                        output += text
                        if visitor_text is not None:
                            visitor_text(text, cm_matrix, tm_matrix, cmap[3], font_size)
                        text = ""
                    text = x + text
                else:  # left-to-right
                    # print(">",xx,x,end="")
                    if rtl_dir:
                        rtl_dir = False
                        output += text
                        if visitor_text is not None:
                            visitor_text(text, cm_matrix, tm_matrix, cmap[3], font_size)
                        text = ""
                    text = text + x
                # fmt: on
    return text, rtl_dir


def extract_page(
    obj: Any,
    pdf: Any,
    orientations: Tuple[int, ...] = (0, 90, 180, 270),
    space_width: float = 200.0,
    content_key: Optional[str] = PG.CONTENTS,
    visitor_operand_before: Optional[Callable[[Any, Any, Any, Any], None]] = None,
    visitor_operand_after: Optional[Callable[[Any, Any, Any, Any], None]] = None,
    visitor_text: Optional[Callable[[Any, Any, Any, Any, Any], None]] = None,
) -> str:
    """
    See extract_text for most arguments.

    Args:
        content_key: indicate the default key where to extract data
            None = the object; this allow to reuse the function on XObject
            default = "/Content"
    """
    text: str = ""
    output: str = ""
    rtl_dir: bool = False  # right-to-left
    cmaps: Dict[
        str,
        Tuple[str, float, Union[str, Dict[int, str]], Dict[str, str], DictionaryObject],
    ] = {}
    try:
        objr = obj
        while NameObject(PG.RESOURCES) not in objr:
            # /Resources can be inherited sometimes so we look to parents
            objr = objr["/Parent"].get_object()
            # if no parents we will have no /Resources will be available
            # => an exception will be raised
        resources_dict = cast(DictionaryObject, objr[PG.RESOURCES])
    except Exception:
        # no resources means no text is possible (no font) we consider the
        # file as not damaged, no need to check for TJ or Tj
        return ""
    if "/Font" in resources_dict:
        for f in cast(DictionaryObject, resources_dict["/Font"]):
            cmaps[f] = build_char_map(f, space_width, obj)
    cmap: Tuple[Union[str, Dict[int, str]], Dict[str, str], str, Optional[DictionaryObject]] = (
        "charmap",
        {},
        "NotInitialized",
        None,
    )  # (encoding,CMAP,font resource name,dictionary-object of font)
    try:
        content = obj[content_key].get_object() if isinstance(content_key, str) else obj
        if not isinstance(content, ContentStream):
            content = ContentStream(content, pdf, "bytes")
    except KeyError:  # it means no content can be extracted(certainly empty page)
        return ""
    # Note: we check all strings are TextStringObjects. ByteStringObjects
    # are strings where the byte->string encoding was unknown, so adding
    # them to the text here would be gibberish.

    cm_matrix: List[float] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    cm_stack = []
    tm_matrix: List[float] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]

    # cm/tm_prev stores the last modified matrices can be an intermediate position
    cm_prev: List[float] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    tm_prev: List[float] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]

    # memo_cm/tm will be used to store the position at the beginning of building the text
    memo_cm: List[float] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    memo_tm: List[float] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    char_scale = 1.0
    space_scale = 1.0
    _space_width: float = 500.0  # will be set correctly at first Tf
    TL = 0.0
    font_size = 12.0  # init just in case of

    def current_spacewidth() -> float:
        return _space_width / 1000.0

    def process_operation(operator: bytes, operands: List[Any]) -> None:
        nonlocal cm_matrix, cm_stack, tm_matrix, cm_prev, tm_prev, memo_cm, memo_tm
        nonlocal char_scale, space_scale, _space_width, TL, font_size, cmap
        nonlocal orientations, rtl_dir, visitor_text, output, text
        global CUSTOM_RTL_MIN, CUSTOM_RTL_MAX, CUSTOM_RTL_SPECIAL_CHARS

        check_crlf_space: bool = False
        # Table 5.4 page 405
        if operator == b"BT":
            tm_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            output += text
            if visitor_text is not None:
                visitor_text(text, memo_cm, memo_tm, cmap[3], font_size)
            text = ""
            memo_cm = cm_matrix.copy()
            memo_tm = tm_matrix.copy()
            return None
        elif operator == b"ET":
            output += text
            if visitor_text is not None:
                visitor_text(text, memo_cm, memo_tm, cmap[3], font_size)
            text = ""
            memo_cm = cm_matrix.copy()
            memo_tm = tm_matrix.copy()
        # table 4.7 "Graphics state operators", page 219
        # cm_matrix calculation is a reserved for the moment
        elif operator == b"q":
            cm_stack.append(
                (
                    cm_matrix,
                    cmap,
                    font_size,
                    char_scale,
                    space_scale,
                    _space_width,
                    TL,
                )
            )
        elif operator == b"Q":
            try:
                (
                    cm_matrix,
                    cmap,
                    font_size,
                    char_scale,
                    space_scale,
                    _space_width,
                    TL,
                ) = cm_stack.pop()
            except Exception:
                cm_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        elif operator == b"cm":
            output += text
            if visitor_text is not None:
                visitor_text(text, memo_cm, memo_tm, cmap[3], font_size)
            text = ""
            cm_matrix = mult(
                [
                    float(operands[0]),
                    float(operands[1]),
                    float(operands[2]),
                    float(operands[3]),
                    float(operands[4]),
                    float(operands[5]),
                ],
                cm_matrix,
            )
            memo_cm = cm_matrix.copy()
            memo_tm = tm_matrix.copy()
        # Table 5.2 page 398
        elif operator == b"Tz":
            char_scale = float(operands[0]) / 100.0
        elif operator == b"Tw":
            space_scale = 1.0 + float(operands[0])
        elif operator == b"TL":
            TL = float(operands[0])
        elif operator == b"Tf":
            if text != "":
                output += text  # .translate(cmap)
                if visitor_text is not None:
                    visitor_text(text, memo_cm, memo_tm, cmap[3], font_size)
            text = ""
            memo_cm = cm_matrix.copy()
            memo_tm = tm_matrix.copy()
            try:
                # charMapTuple: font_type, float(sp_width / 2), encoding,
                #               map_dict, font-dictionary
                charMapTuple = cmaps[operands[0]]
                _space_width = charMapTuple[1]
                # current cmap: encoding, map_dict, font resource name
                #               (internal name, not the real font-name),
                # font-dictionary. The font-dictionary describes the font.
                cmap = (
                    charMapTuple[2],
                    charMapTuple[3],
                    operands[0],
                    charMapTuple[4],
                )
            except KeyError:  # font not found
                _space_width = unknown_char_map[1]
                cmap = (
                    unknown_char_map[2],
                    unknown_char_map[3],
                    "???" + operands[0],
                    None,
                )
            try:
                font_size = float(operands[1])
            except Exception:
                pass  # keep previous size
        # Table 5.5 page 406
        elif operator == b"Td":
            check_crlf_space = True
            # A special case is a translating only tm:
            # tm[0..5] = 1 0 0 1 e f,
            # i.e. tm[4] += tx, tm[5] += ty.
            tx = float(operands[0])
            ty = float(operands[1])
            tm_matrix[4] += tx * tm_matrix[0] + ty * tm_matrix[2]
            tm_matrix[5] += tx * tm_matrix[1] + ty * tm_matrix[3]
        elif operator == b"Tm":
            check_crlf_space = True
            tm_matrix = [
                float(operands[0]),
                float(operands[1]),
                float(operands[2]),
                float(operands[3]),
                float(operands[4]),
                float(operands[5]),
            ]
        elif operator == b"T*":
            check_crlf_space = True
            tm_matrix[5] -= TL

        elif operator == b"Tj":
            check_crlf_space = True
            text, rtl_dir = handle_tj(
                text,
                operands,
                cm_matrix,
                tm_matrix,  # text matrix
                cmap,
                orientations,
                output,
                font_size,
                rtl_dir,
                visitor_text,
            )
        else:
            return None
        if check_crlf_space:
            try:
                text, output, cm_prev, tm_prev = crlf_space_check(
                    text,
                    (cm_prev, tm_prev),
                    (cm_matrix, tm_matrix),
                    (memo_cm, memo_tm),
                    cmap,
                    orientations,
                    output,
                    font_size,
                    visitor_text,
                    current_spacewidth(),
                )
                if text == "":
                    memo_cm = cm_matrix.copy()
                    memo_tm = tm_matrix.copy()
            except OrientationNotFoundError:
                return None

    for operands, operator in content.operations:
        if visitor_operand_before is not None:
            visitor_operand_before(operator, operands, cm_matrix, tm_matrix)
        # multiple operators are defined in here ####
        if operator == b"'":
            process_operation(b"T*", [])
            process_operation(b"Tj", operands)
        elif operator == b'"':
            process_operation(b"Tw", [operands[0]])
            process_operation(b"Tc", [operands[1]])
            process_operation(b"T*", [])
            process_operation(b"Tj", operands[2:])
        elif operator == b"TD":
            process_operation(b"TL", [-operands[1]])
            process_operation(b"Td", operands)
        elif operator == b"TJ":
            for op in operands[0]:
                if isinstance(op, (str, bytes)):
                    process_operation(b"Tj", [op])
                if isinstance(op, (int, float, NumberObject, FloatObject)) and ((abs(float(op)) >= _space_width) and (len(text) > 0) and (text[-1] != " ")):
                    process_operation(b"Tj", [" "])
        elif operator == b"Do":
            output += text
            if visitor_text is not None:
                visitor_text(text, memo_cm, memo_tm, cmap[3], font_size)
            try:
                if output[-1] != "\n":
                    output += "\n"
                    if visitor_text is not None:
                        visitor_text(
                            "\n",
                            memo_cm,
                            memo_tm,
                            cmap[3],
                            font_size,
                        )
            except IndexError:
                pass
            try:
                xobj = resources_dict["/XObject"]
                if xobj[operands[0]]["/Subtype"] != "/Image":  # type: ignore
                    text = self.extract_xform_text(
                        xobj[operands[0]],  # type: ignore
                        orientations,
                        space_width,
                        visitor_operand_before,
                        visitor_operand_after,
                        visitor_text,
                    )
                    output += text
                    if visitor_text is not None:
                        visitor_text(
                            text,
                            memo_cm,
                            memo_tm,
                            cmap[3],
                            font_size,
                        )
            except Exception:
                print(
                    f" impossible to decode XFormObject {operands[0]}",
                    __name__,
                )
            finally:
                text = ""
                memo_cm = cm_matrix.copy()
                memo_tm = tm_matrix.copy()

        else:
            process_operation(operator, operands)
        if visitor_operand_after is not None:
            visitor_operand_after(operator, operands, cm_matrix, tm_matrix)
    output += text  # just in case of
    if text != "" and visitor_text is not None:
        visitor_text(text, memo_cm, memo_tm, cmap[3], font_size)
    return output
