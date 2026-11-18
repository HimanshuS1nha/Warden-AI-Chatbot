"""Turn a policy document (PDF, DOCX or Markdown) into a list of ParsedBlocks.

A block is one paragraph, list item or table, in reading order, with the
heading path it sits under. Headings themselves are not blocks; they only set
`heading_path` for the blocks after them. Tables become Markdown tables so a
row's cells stay together through chunking.

The parser keeps text as it is on the page, with line breaks. `cleaning`
joins wrapped lines, fixes hyphenation and drops running headers and footers.

PDF headings are found from fonts, since PDFs have no heading markup: a
heading is a bold paragraph larger than the body text, and each such font
size is a heading level (larger is higher). A size used by only one paragraph
in the whole document is a title or banner, not a section heading, and is
treated as text.
"""

import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import docx
import pymupdf
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph


class BlockKind(StrEnum):
    TEXT = "text"
    TABLE = "table"


@dataclass(frozen=True, slots=True)
class ParsedBlock:
    text: str
    # 1-based page number. None for DOCX and Markdown, which have no fixed pages.
    page: int | None
    heading_path: tuple[str, ...]
    kind: BlockKind = BlockKind.TEXT
    # True for PDF text in the top or bottom margin of the page, where running
    # headers and footers sit.
    in_margin: bool = False


class UnsupportedDocumentError(ValueError):
    pass


def parse_document(path: Path) -> list[ParsedBlock]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix == ".docx":
        return parse_docx(path)
    if suffix in {".md", ".markdown"}:
        return parse_markdown(path.read_text(encoding="utf-8"))
    raise UnsupportedDocumentError(f"Unsupported document type: {path.name}")


def _with_heading(path: tuple[str, ...], level: int, heading: str) -> tuple[str, ...]:
    return (*path[: level - 1], heading)


def join_cell_lines(lines: list[str]) -> str:
    """Join the wrapped lines of one table cell. A line ending in a hyphen
    joins without a space and keeps the hyphen ("ride-" + "hailing")."""
    joined = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if joined and not joined.endswith("-"):
            joined += " "
        joined += line
    return joined


def markdown_table(rows: list[list[str]]) -> str:
    """Render rows as a Markdown table. The first row is the header."""
    width = max(len(row) for row in rows)

    def render(row: list[str]) -> str:
        cells = [cell.replace("|", "\\|") for cell in row] + [""] * (width - len(row))
        return "| " + " | ".join(cells) + " |"

    lines = [render(rows[0]), "|" + "---|" * width]
    lines.extend(render(row) for row in rows[1:])
    return "\n".join(lines)


# PDF ------------------------------------------------------------------------

# Running headers and footers sit in the top and bottom 7% of the page.
MARGIN_FRACTION = 0.07
# A heading is at least this much larger than the body text.
HEADING_SIZE_RATIO = 1.08
# Two lines are the same paragraph when the gap between them is smaller than
# this share of the font size.
PARAGRAPH_GAP_RATIO = 0.6
# Table lines more than this many points below the row above start a new row.
TABLE_ROW_GAP = 3.0

BULLET_MARKER = re.compile(r"^[•●▪◦‣∙·–-]$")
NUMBER_MARKER = re.compile(r"^(\d{1,2}|[a-z])[.)]$")


@dataclass(slots=True)
class _Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    bold: bool
    marker: str | None = None


@dataclass(slots=True)
class _Paragraph:
    page: int
    lines: list[_Line]
    in_margin: bool
    # Set when this paragraph stands for a table: the Markdown of the table.
    table: str | None = None

    @property
    def size(self) -> float:
        return round(self.lines[0].size, 1) if self.lines else 0.0

    @property
    def bold(self) -> bool:
        return all(line.bold for line in self.lines)


def parse_pdf(path: Path) -> list[ParsedBlock]:
    with pymupdf.open(path) as doc:
        pages = [_parse_pdf_page(page, number) for number, page in enumerate(doc, start=1)]

    paragraphs = [para for page in pages for para in page]
    body_size = _body_font_size(paragraphs)
    levels = _heading_levels(paragraphs, body_size)

    blocks: list[ParsedBlock] = []
    heading_path: tuple[str, ...] = ()
    for para in paragraphs:
        if para.table is not None:
            blocks.append(
                ParsedBlock(para.table, para.page, heading_path, BlockKind.TABLE, para.in_margin)
            )
            continue
        level = levels.get(para.size) if para.bold and not para.in_margin else None
        if level is not None:
            heading = " ".join(line.text for line in para.lines)
            heading_path = _with_heading(heading_path, level, heading)
            continue
        text = "\n".join(line.text for line in para.lines)
        marker = para.lines[0].marker
        if marker is not None:
            text = f"{marker} {text}"
        blocks.append(ParsedBlock(text, para.page, heading_path, BlockKind.TEXT, para.in_margin))
    return blocks


def _parse_pdf_page(page: pymupdf.Page, number: int) -> list[_Paragraph]:
    tables = list(page.find_tables().tables)
    table_boxes = [pymupdf.Rect(table.bbox) for table in tables]
    lines = _attach_list_markers(_page_lines(page))

    height = page.rect.height
    top, bottom = height * MARGIN_FRACTION, height * (1 - MARGIN_FRACTION)

    paragraphs: list[_Paragraph] = []
    # Each table is one paragraph, placed where its first line falls in
    # reading order and filled in once all its lines are known.
    table_paragraphs: dict[int, tuple[_Paragraph, list[_Line]]] = {}
    current: _Paragraph | None = None
    for line in lines:
        center = pymupdf.Point((line.x0 + line.x1) / 2, (line.y0 + line.y1) / 2)
        table_index = next((i for i, box in enumerate(table_boxes) if center in box), None)
        if table_index is not None:
            if table_index not in table_paragraphs:
                placeholder = _Paragraph(number, [], in_margin=False, table="")
                table_paragraphs[table_index] = (placeholder, [])
                paragraphs.append(placeholder)
                current = None
            table_paragraphs[table_index][1].append(line)
            continue

        if current is not None and not _starts_paragraph(current, line):
            current.lines.append(line)
            continue
        current = _Paragraph(number, [line], in_margin=line.y0 < top or line.y1 > bottom)
        paragraphs.append(current)

    for index, (placeholder, table_lines) in table_paragraphs.items():
        placeholder.table = _pdf_table_markdown(tables[index], table_lines)
    return paragraphs


def _page_lines(page: pymupdf.Page) -> list[_Line]:
    """Lines in PyMuPDF's reading order, with their font size and boldness."""
    lines: list[_Line] = []
    data: dict[str, Any] = page.get_text("dict")
    for block in data["blocks"]:
        if block["type"] != 0:  # 0 is text, 1 is image
            continue
        for line in block["lines"]:
            spans = [span for span in line["spans"] if span["text"].strip()]
            if not spans:
                continue
            text = "".join(span["text"] for span in line["spans"]).strip()
            # The size of the span with the most text, so a small footnote
            # marker doesn't decide the size of the line.
            main = max(spans, key=lambda span: len(span["text"]))
            x0, y0, x1, y1 = line["bbox"]
            lines.append(
                _Line(
                    text=text,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    size=main["size"],
                    bold=all(span["flags"] & pymupdf.TEXT_FONT_BOLD for span in spans),
                )
            )
    return lines


def _attach_list_markers(lines: list[_Line]) -> list[_Line]:
    """PyMuPDF often returns list bullets and numbers as separate lines after
    the text. Move each one onto the item it belongs to: the nearest line to
    its right at the same height."""
    position = {id(line): i for i, line in enumerate(lines)}
    markers = [line for line in lines if _marker_text(line.text) is not None]
    rest = [line for line in lines if _marker_text(line.text) is None]
    for marker in markers:
        candidates = [
            line
            for line in rest
            if abs(line.y0 - marker.y0) < 2 and line.x0 > marker.x0 and line.marker is None
        ]
        if not candidates:
            # A lone "-" or "1." with nothing beside it is ordinary text.
            rest.append(marker)
            continue
        item = min(candidates, key=lambda line: line.x0 - marker.x0)
        item.marker = _marker_text(marker.text)
    return sorted(rest, key=lambda line: position[id(line)])


def _marker_text(text: str) -> str | None:
    text = text.strip()
    if BULLET_MARKER.match(text):
        return "-"
    if NUMBER_MARKER.match(text):
        return text
    return None


def _starts_paragraph(para: _Paragraph, line: _Line) -> bool:
    previous = para.lines[-1]
    if line.marker is not None:
        return True
    if abs(line.size - previous.size) > 0.3:
        return True
    gap = line.y0 - previous.y1
    # Negative gap: the line is beside or above the previous one (another
    # column, or a header line placed out of order).
    if gap < -previous.size * 0.5 or gap > previous.size * PARAGRAPH_GAP_RATIO:
        return True
    # Wrapped lines start where the paragraph's first line starts.
    return abs(line.x0 - para.lines[0].x0) > 2


def _body_font_size(paragraphs: list[_Paragraph]) -> float:
    sizes: Counter[float] = Counter()
    for para in paragraphs:
        for line in para.lines:
            sizes[round(line.size, 1)] += len(line.text)
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _heading_levels(paragraphs: list[_Paragraph], body_size: float) -> dict[float, int]:
    counts = Counter(
        para.size
        for para in paragraphs
        if para.table is None
        and para.bold
        and not para.in_margin
        and para.size >= body_size * HEADING_SIZE_RATIO
    )
    sizes = sorted((size for size, count in counts.items() if count >= 2), reverse=True)
    return {size: level for level, size in enumerate(sizes, start=1)}


def _pdf_table_markdown(table: Any, lines: list[_Line]) -> str:
    """Rebuild a table from its lines: columns from the cells PyMuPDF found,
    rows from vertical gaps. PyMuPDF's own rows split a cell's wrapped lines
    into separate rows, and its cell text loses ligatures like "fi"."""
    column_starts: list[float] = []
    for x0 in sorted(cell[0] for cell in table.cells if cell is not None):
        if not column_starts or x0 - column_starts[-1] > 2:
            column_starts.append(x0)

    def column(line: _Line) -> int:
        return max(
            (i for i, start in enumerate(column_starts) if line.x0 >= start - 1),
            default=0,
        )

    rows: list[list[_Line]] = []
    row_bottom = float("-inf")
    for line in sorted(lines, key=lambda line: (line.y0, line.x0)):
        if not rows or line.y0 > row_bottom + TABLE_ROW_GAP:
            rows.append([])
            row_bottom = line.y1
        rows[-1].append(line)
        row_bottom = max(row_bottom, line.y1)

    cells_by_row: list[list[str]] = []
    for row in rows:
        cells: list[list[str]] = [[] for _ in column_starts]
        for line in sorted(row, key=lambda line: line.y0):
            cells[column(line)].append(line.text)
        cells_by_row.append([join_cell_lines(cell) for cell in cells])
    return markdown_table(cells_by_row)


# DOCX -----------------------------------------------------------------------

DOCX_HEADING_STYLE = re.compile(r"^Heading (\d)$")


def parse_docx(path: Path) -> list[ParsedBlock]:
    document = docx.Document(str(path))
    blocks: list[ParsedBlock] = []
    heading_path: tuple[str, ...] = ()
    for item in document.iter_inner_content():
        if isinstance(item, DocxTable):
            rows = _docx_table_rows(item)
            if rows:
                blocks.append(
                    ParsedBlock(markdown_table(rows), None, heading_path, BlockKind.TABLE)
                )
            continue
        text = item.text.strip()
        if not text:
            continue
        style = _docx_style_name(item)
        match = DOCX_HEADING_STYLE.match(style)
        if match:
            heading_path = _with_heading(heading_path, int(match.group(1)), text)
            continue
        if style.startswith("List"):
            text = f"- {text}"
        blocks.append(ParsedBlock(text, None, heading_path))
    return blocks


def _docx_style_name(paragraph: DocxParagraph) -> str:
    style = paragraph.style
    name = style.name if style is not None else None
    return str(name) if name else ""


def _docx_table_rows(table: DocxTable) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table.rows:
        cells: list[str] = []
        previous: object = None
        for cell in row.cells:
            # A merged cell appears once per grid column it spans.
            if cell._tc is previous:
                continue
            previous = cell._tc
            cells.append(join_cell_lines(cell.text.splitlines()))
        if any(cells):
            rows.append(cells)
    return rows


# Markdown -------------------------------------------------------------------

MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
MD_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


def parse_markdown(text: str) -> list[ParsedBlock]:
    return list(_markdown_blocks(text))


def _markdown_blocks(text: str) -> Iterator[ParsedBlock]:
    heading_path: tuple[str, ...] = ()
    buffer: list[str] = []

    def flush() -> Iterator[ParsedBlock]:
        if buffer:
            kind = (
                BlockKind.TABLE
                if all(line.lstrip().startswith("|") for line in buffer)
                else BlockKind.TEXT
            )
            yield ParsedBlock("\n".join(buffer), None, heading_path, kind)
            buffer.clear()

    for raw in text.splitlines():
        line = raw.rstrip()
        heading = MD_HEADING.match(line)
        if heading:
            yield from flush()
            heading_path = _with_heading(heading_path, len(heading.group(1)), heading.group(2))
            continue
        if not line.strip():
            yield from flush()
            continue
        is_table_line = line.lstrip().startswith("|")
        buffer_is_table = bool(buffer) and buffer[0].lstrip().startswith("|")
        # Each list item is its own block, and a table never shares a block
        # with text.
        if buffer and (MD_LIST_ITEM.match(line) or is_table_line != buffer_is_table):
            yield from flush()
        buffer.append(line.strip())
    yield from flush()
