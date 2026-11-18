from pathlib import Path

import docx
import pytest

from warden.rag.cleaning import clean_blocks
from warden.rag.parsing import (
    BlockKind,
    ParsedBlock,
    UnsupportedDocumentError,
    markdown_table,
    parse_document,
    parse_markdown,
)

DOCUMENTS = Path(__file__).resolve().parents[2] / "documents"


def test_markdown_headings_set_the_heading_path() -> None:
    blocks = parse_markdown(
        "# Leave Policy\n\nIntro.\n\n## 3. Entitlement\n\n### 3.1 Joining\n\nPro-rated.\n\n"
        "## 4. Counting\n\nWorking days.\n"
    )
    assert [(b.heading_path, b.text) for b in blocks] == [
        (("Leave Policy",), "Intro."),
        (("Leave Policy", "3. Entitlement", "3.1 Joining"), "Pro-rated."),
        (("Leave Policy", "4. Counting"), "Working days."),
    ]
    assert all(b.page is None for b in blocks)


def test_markdown_list_items_and_tables_are_separate_blocks() -> None:
    blocks = parse_markdown(
        "Rules:\n- Whole days only.\n- One leave year\n  per request.\n"
        "| Leave | Days |\n|---|---|\n| Casual | 12 |\n"
    )
    assert [(b.kind, b.text) for b in blocks] == [
        (BlockKind.TEXT, "Rules:"),
        (BlockKind.TEXT, "- Whole days only."),
        (BlockKind.TEXT, "- One leave year\nper request."),
        (BlockKind.TABLE, "| Leave | Days |\n|---|---|\n| Casual | 12 |"),
    ]


def test_markdown_table_escapes_pipes_and_pads_short_rows() -> None:
    assert markdown_table([["Type", "Notes"], ["A|B"]]) == (
        "| Type | Notes |\n|---|---|\n| A\\|B |  |"
    )


def test_docx_headings_lists_and_tables(tmp_path: Path) -> None:
    document = docx.Document()
    document.add_heading("Travel Policy", level=0)  # Title style, not a heading level
    document.add_heading("1. Limits", level=1)
    document.add_paragraph("Book through Finance.")
    document.add_paragraph("Economy class only.", style="List Bullet")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Expense", "Limit"
    table.cell(1, 0).text, table.cell(1, 1).text = "Hotel", "INR 6,000"
    document.add_heading("1.1 Advances", level=2)
    document.add_paragraph("Up to 70%.")
    path = tmp_path / "travel.docx"
    document.save(str(path))

    assert parse_document(path) == [
        ParsedBlock("Travel Policy", None, ()),
        ParsedBlock("Book through Finance.", None, ("1. Limits",)),
        ParsedBlock("- Economy class only.", None, ("1. Limits",)),
        ParsedBlock(
            "| Expense | Limit |\n|---|---|\n| Hotel | INR 6,000 |",
            None,
            ("1. Limits",),
            BlockKind.TABLE,
        ),
        ParsedBlock("Up to 70%.", None, ("1. Limits", "1.1 Advances")),
    ]


def test_unsupported_file_type(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(UnsupportedDocumentError):
        parse_document(path)


# The real corpus: checks that the PDF heuristics hold on the documents we ship.


@pytest.fixture(scope="module")
def leave_policy() -> list[ParsedBlock]:
    return clean_blocks(parse_document(DOCUMENTS / "leave-policy-v2.pdf"))


def test_pdf_heading_levels_come_from_font_sizes(leave_policy: list[ParsedBlock]) -> None:
    paths = {b.heading_path for b in leave_policy}
    assert ("1. Purpose",) in paths
    assert ("3. Leave year and entitlement", "3.1 Joining during the year") in paths
    assert ("14. Revision history",) in paths
    # The title appears once, so it's text, not a heading over the whole document.
    assert not any(path and path[0] == "Leave Policy" for path in paths)


def test_pdf_tables_keep_wrapped_cells_in_one_row(leave_policy: list[ParsedBlock]) -> None:
    table = next(b for b in leave_policy if b.kind is BlockKind.TABLE)
    assert table.heading_path == ("3. Leave year and entitlement",)
    assert table.page == 1
    rows = table.text.split("\n")
    assert rows[0] == (
        "| Leave type | Annual entitlement | Carry forward to next year "
        "| Notice expected | Supporting documents |"
    )
    assert len(rows) == 5  # header, separator, casual, sick, earned
    assert rows[4] == (
        "| Earned leave | 18 working days | Up to 8 days "
        "| 14 calendar days for requests of 3 or more days | None |"
    )
    # PyMuPDF's own table text drops the "fi" ligature; ours keeps it.
    assert "Medical certificate" in table.text


def test_pdf_list_markers_join_their_items(leave_policy: list[ParsedBlock]) -> None:
    texts = [b.text for b in leave_policy]
    assert "4. You have enough available balance for the full request." in texts
    assert any(t.startswith("- Sick leave can") or t.startswith("- You may apply") for t in texts)
    assert "•" not in " ".join(texts)


def test_pdf_running_headers_and_footers_are_removed(leave_policy: list[ParsedBlock]) -> None:
    text = "\n".join(b.text for b in leave_policy)
    assert "Page 1 of 4" not in text
    assert "Warden · Leave Policy · Version 2" not in text
    assert "WD-HR-002 · Leave Policy" not in text
    assert "W A R D E N" not in text


def test_pdf_table_split_across_pages_is_one_table_after_cleaning() -> None:
    path = DOCUMENTS / "travel-reimbursement-policy.pdf"
    section = ("4. Travel and accommodation limits",)

    def limit_tables(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
        return [b for b in blocks if b.kind is BlockKind.TABLE and b.heading_path == section]

    parsed = limit_tables(parse_document(path))
    # PyMuPDF sees two tables, each with the header row.
    assert [b.page for b in parsed] == [1, 2]
    assert all(b.text.startswith("| Expense | Band A | Band B |") for b in parsed)

    [table] = limit_tables(clean_blocks(parse_document(path)))
    rows = table.text.split("\n")
    assert table.page == 1
    assert rows.count("| Expense | Band A | Band B |") == 1
    assert rows[2].startswith("| Domestic flights |")
    assert rows[-1].startswith("| Local transport | Actual cost of taxi, metro or ride-hailing |")
