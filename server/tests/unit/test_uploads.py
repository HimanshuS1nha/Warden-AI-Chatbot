import io
import zipfile

import docx
import pytest

from warden.services.errors import FileTooLarge, InvalidUpload
from warden.services.uploads import MAX_UPLOAD_BYTES, FileKind, check_upload

PDF = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
MARKDOWN = b"# Leave Policy\n\n## Carry forward\n\nUp to 8 days.\n"


def make_docx() -> bytes:
    document = docx.Document()
    document.add_paragraph("Up to 8 days carry forward.")
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def make_zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("filename", "data", "kind"),
    [
        ("policy.pdf", PDF, FileKind.PDF),
        ("POLICY.PDF", PDF, FileKind.PDF),
        ("policy.docx", make_docx(), FileKind.DOCX),
        ("policy.md", MARKDOWN, FileKind.MARKDOWN),
        ("policy.markdown", MARKDOWN, FileKind.MARKDOWN),
    ],
)
def test_accepts_pdf_docx_and_markdown(filename: str, data: bytes, kind: FileKind) -> None:
    assert check_upload(filename, data) is kind


@pytest.mark.parametrize(
    ("filename", "data"),
    [
        # Extension and content disagree.
        ("policy.pdf", MARKDOWN),
        ("policy.pdf", b"\x89PNG\r\n\x1a\n" + b"\x00" * 32),
        ("policy.docx", PDF),
        ("policy.md", PDF),
        ("policy.md", make_docx()),
        # A ZIP that isn't a Word document.
        ("policy.docx", make_zip({"hello.txt": b"hi"})),
        ("policy.docx", b"PK\x03\x04 not really a zip"),
        # Not text.
        ("policy.md", b"# Title\x00\x01"),
        ("policy.md", "# Café".encode("latin-1")),
        # Types that aren't accepted at all, whatever the content.
        ("policy.txt", MARKDOWN),
        ("policy.html", b"<h1>Policy</h1>"),
        ("policy", PDF),
        (None, PDF),
        # Empty.
        ("policy.pdf", b""),
    ],
)
def test_refuses_anything_else(filename: str | None, data: bytes) -> None:
    with pytest.raises(InvalidUpload):
        check_upload(filename, data)


def test_refuses_a_docx_that_unpacks_too_large() -> None:
    # Zeros compress to almost nothing: under a megabyte packed, 60 MB unpacked.
    bomb = make_zip(
        {
            "[Content_Types].xml": b"<Types/>",
            "word/document.xml": b"<w:document/>",
            "word/media/padding.bin": bytes(60 * 1024 * 1024),
        }
    )
    assert len(bomb) < 1024 * 1024

    with pytest.raises(InvalidUpload):
        check_upload("policy.docx", bomb)


def test_refuses_files_over_the_limit() -> None:
    check_upload("policy.pdf", PDF + bytes(MAX_UPLOAD_BYTES - len(PDF)))

    with pytest.raises(FileTooLarge):
        check_upload("policy.pdf", PDF + bytes(MAX_UPLOAD_BYTES - len(PDF) + 1))
