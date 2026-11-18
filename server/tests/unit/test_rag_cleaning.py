from warden.rag.cleaning import (
    clean_blocks,
    fix_hyphenation,
    furniture_key,
    normalize_whitespace,
    unwrap_lines,
    word_vocabulary,
)
from warden.rag.parsing import BlockKind, ParsedBlock


def test_normalize_whitespace_collapses_spaces_and_odd_characters() -> None:
    text = "  Casual leave\tis   12 days.​ \nSick­leave  \n\n\n\nEnd  "
    assert normalize_whitespace(text) == "Casual leave is 12 days.\nSickleave\n\nEnd"


def test_normalize_whitespace_converts_windows_line_endings() -> None:
    assert normalize_whitespace("one\r\ntwo\rthree") == "one\ntwo\nthree"


def test_hyphen_dropped_when_joined_word_is_known() -> None:
    vocabulary = word_vocabulary(["Send a medical certificate."])
    assert fix_hyphenation("a medi-\ncal certificate", vocabulary) == "a medical certificate"


def test_hyphen_kept_for_real_compound_words() -> None:
    vocabulary = word_vocabulary(["taxi, metro or ride-hailing"])
    assert fix_hyphenation("metro or ride-\nhailing", vocabulary) == "metro or ride-hailing"


def test_hyphen_inside_a_line_is_left_alone() -> None:
    assert fix_hyphenation("full-time staff", {"fulltime"}) == "full-time staff"


def test_unwrap_lines_joins_with_spaces() -> None:
    assert unwrap_lines("The leave year runs\nfrom 1 January\n\nto 31 December.") == (
        "The leave year runs from 1 January to 31 December."
    )


def test_furniture_key_ignores_spacing_and_numbers_but_not_case() -> None:
    assert furniture_key("Page 1 of 4") == furniture_key("Page 3 of 4")
    assert furniture_key("W A R D E N") == furniture_key("WA R D E N")
    assert furniture_key("INTERNAL") != furniture_key("Internal")


def _block(
    text: str, page: int = 1, *, in_margin: bool = False, kind: BlockKind = BlockKind.TEXT
) -> ParsedBlock:
    return ParsedBlock(text, page, ("1. Purpose",), kind, in_margin)


def test_clean_blocks_drops_margin_blocks_and_body_blocks_that_repeat_them() -> None:
    blocks = [
        _block("W A R D E N"),  # cover banner, below the margin on page 1
        _block("Leave Policy"),
        _block("This policy sets out\nthe paid leave."),
        _block("Warden · Leave Policy · Version 2", in_margin=True),
        _block("Page 1 of 2", in_margin=True),
        _block("WA R D E N", page=2, in_margin=True),
        _block("Leave is counted in\nworking days.", page=2),
        _block("Page 2 of 2", page=2, in_margin=True),
    ]
    assert [b.text for b in clean_blocks(blocks)] == [
        "Leave Policy",
        "This policy sets out the paid leave.",
        "Leave is counted in working days.",
    ]


def test_clean_blocks_keeps_long_body_text_that_matches_a_header() -> None:
    header = "a b c d e f g h i"
    blocks = [_block(header, in_margin=True), _block(header, page=2)]
    assert [b.text for b in clean_blocks(blocks)] == [header]


def test_clean_blocks_keeps_table_rows_on_separate_lines() -> None:
    table = "| Leave | Days |\n|---|---|\n| Casual | 12 |"
    blocks = [_block(table, kind=BlockKind.TABLE)]
    assert clean_blocks(blocks)[0].text == table


def test_clean_blocks_uses_the_whole_document_as_hyphenation_vocabulary() -> None:
    blocks = [
        _block("Attach a medi-\ncal certificate."),
        _block("Medical appointments count as sick leave.", page=2),
    ]
    assert clean_blocks(blocks)[0].text == "Attach a medical certificate."


def test_clean_blocks_drops_empty_blocks() -> None:
    assert clean_blocks([_block(" ​ \n ")]) == []


def test_table_continued_on_the_next_page_is_merged() -> None:
    header = "| Expense | Band A |\n|---|---|"
    blocks = [
        _block(header + "\n| Flights | Economy |", kind=BlockKind.TABLE),
        _block("Page 1 of 2", in_margin=True),
        _block(header + "\n| Hotel | INR 6,000 |", page=2, kind=BlockKind.TABLE),
        _block("Metro cities are listed below.", page=2),
    ]
    cleaned = clean_blocks(blocks)
    assert [(b.kind, b.page, b.text) for b in cleaned] == [
        (BlockKind.TABLE, 1, header + "\n| Flights | Economy |\n| Hotel | INR 6,000 |"),
        (BlockKind.TEXT, 2, "Metro cities are listed below."),
    ]


def test_different_tables_are_not_merged() -> None:
    blocks = [
        _block("| A | B |\n|---|---|\n| 1 | 2 |", kind=BlockKind.TABLE),
        _block("| C | D |\n|---|---|\n| 3 | 4 |", kind=BlockKind.TABLE),
    ]
    assert len(clean_blocks(blocks)) == 2
