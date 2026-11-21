"""Checks on uploaded policy documents.

Only PDF, DOCX and Markdown are accepted, up to MAX_UPLOAD_BYTES. The type
comes from the file's content, not its name or the client's Content-Type:
the extension has to agree with what the bytes are, or the upload is
refused.

- PDF: starts with "%PDF-".
- DOCX: a ZIP archive holding word/document.xml. Its unpacked size is
  capped too, so a small archive can't unpack into gigabytes when parsed.
- Markdown: has no signature, so it must be UTF-8 text with no NUL bytes
  that doesn't start with another type's signature (a renamed PDF is
  mostly ASCII).
"""

import io
import zipfile
from enum import StrEnum
from pathlib import PurePath

from warden.services.errors import FileTooLarge, InvalidUpload

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_DOCX_UNPACKED_BYTES = 50 * 1024 * 1024


class FileKind(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"

    @property
    def suffix(self) -> str:
        """The extension the file is saved with, which picks its parser."""
        return {FileKind.PDF: ".pdf", FileKind.DOCX: ".docx", FileKind.MARKDOWN: ".md"}[self]


_BY_EXTENSION = {
    ".pdf": FileKind.PDF,
    ".docx": FileKind.DOCX,
    ".md": FileKind.MARKDOWN,
    ".markdown": FileKind.MARKDOWN,
}


def check_upload(filename: str | None, data: bytes) -> FileKind:
    """The file's kind, or InvalidUpload / FileTooLarge."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise FileTooLarge(
            f"Files can be at most {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
            max_bytes=MAX_UPLOAD_BYTES,
        )
    if not data:
        raise InvalidUpload("The file is empty")

    extension = PurePath(filename or "").suffix.lower()
    kind = _BY_EXTENSION.get(extension)
    if kind is None:
        raise InvalidUpload("Only PDF, DOCX and Markdown (.md) files are accepted")
    if not _content_is(kind, data):
        raise InvalidUpload(f"The file's content isn't a valid {extension} file")
    return kind


def _content_is(kind: FileKind, data: bytes) -> bool:
    if kind is FileKind.PDF:
        return data.startswith(b"%PDF-")
    if kind is FileKind.DOCX:
        return _is_docx(data)
    return _is_text(data)


def _is_docx(data: bytes) -> bool:
    if not data.startswith(b"PK\x03\x04"):
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
    except zipfile.BadZipFile:
        return False
    names = {member.filename for member in members}
    unpacked = sum(member.file_size for member in members)
    return (
        "word/document.xml" in names
        and "[Content_Types].xml" in names
        and unpacked <= MAX_DOCX_UNPACKED_BYTES
    )


def _is_text(data: bytes) -> bool:
    if b"\x00" in data or data.startswith((b"%PDF-", b"PK\x03\x04")):
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True
