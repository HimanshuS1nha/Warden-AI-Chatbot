from pathlib import Path

import pytest
import yaml

from warden.models import AccessLevel
from warden.rag.ingest import ManifestError, load_manifest

SERVER_DIR = Path(__file__).resolve().parents[2]


def entry(**overrides: object) -> dict[str, object]:
    return {
        "slug": "leave-policy",
        "title": "Leave Policy",
        "access_level": "all",
        "version": 1,
        "effective_date": "2026-01-01",
        "file": "leave.md",
    } | overrides


def write(tmp_path: Path, *entries: dict[str, object]) -> Path:
    (tmp_path / "leave.md").write_text("# Leave")
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump({"documents": list(entries)}))
    return path


def test_the_shipped_manifest_is_valid() -> None:
    entries = load_manifest(SERVER_DIR / "documents" / "manifest.yaml")
    assert len(entries) == 13
    levels = {e.slug: e.access_level for e in entries}
    assert levels["manager-handbook"] is AccessLevel.MANAGER
    assert levels["disciplinary-procedure"] is AccessLevel.HR
    assert sorted(e.version for e in entries if e.slug == "leave-policy") == [1, 2]


def test_valid_entry(tmp_path: Path) -> None:
    [loaded] = load_manifest(write(tmp_path, entry()))
    assert loaded.slug == "leave-policy"
    assert loaded.access_level is AccessLevel.ALL


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"access_level": "public"}, "access_level"),
        ({"version": 0}, "version"),
        ({"slug": "Leave Policy"}, "slug"),
        ({"file": "missing.md"}, "doesn't exist"),
        ({"file": "../outside.md"}, "outside"),
        ({"extra": "field"}, "extra"),
    ],
)
def test_invalid_entries(tmp_path: Path, overrides: dict[str, object], message: str) -> None:
    (tmp_path.parent / "outside.md").write_text("x")
    with pytest.raises(ManifestError, match=message):
        load_manifest(write(tmp_path, entry(**overrides)))


def test_duplicate_version(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="listed twice"):
        load_manifest(write(tmp_path, entry(), entry()))


def test_versions_must_agree_on_access_level(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="same title and access level"):
        load_manifest(write(tmp_path, entry(), entry(version=2, access_level="hr")))
