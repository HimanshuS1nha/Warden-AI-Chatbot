"""/api/admin/documents end to end: upload, background indexing into an
in-process Qdrant, reindex and listing.

httpx's ASGI transport waits for the app to finish, background tasks
included, so indexing is done by the time a request returns.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from httpx import AsyncClient, Response
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from tests.factories import MakeEmployee
from tests.integration.test_auth import login
from tests.integration.test_leave_service import audit_rows
from tests.rag_fakes import FakeEmbeddingClient, WhitespaceTokenCounter, no_sleep
from tests.unit.test_uploads import make_docx
from warden.core.dependencies import DbSession
from warden.graph.resources import AgentResources
from warden.main import app
from warden.models import AuditAction, AuditOutcome, EmployeeModel, Role
from warden.rag.embeddings import Embedder
from warden.rag.index import PolicyIndex
from warden.rag.indexer import DocumentIndexer
from warden.routers.admin import get_document_indexer
from warden.routers.chat import get_agent
from warden.routers.services import get_document_service
from warden.services.documents import DocumentService
from warden.services.uploads import MAX_UPLOAD_BYTES

POLICY_V1 = b"""# Remote Work Policy

## Eligibility

Employees can work remotely 2 days a week.
"""

POLICY_V2 = b"""# Remote Work Policy

## Eligibility

Employees can work remotely 3 days a week.

## Equipment

Laptops are provided by the company.
"""

# Parses to nothing, so indexing fails.
BLANK = b"   \n\n  \n"


class Admin:
    """The indexer the routes use, with its parts exposed to the tests."""

    def __init__(self, connection: AsyncConnection, tmp_path: Path, index: PolicyIndex) -> None:
        self.upload_dir = tmp_path / "uploads"
        self.cache_dir = tmp_path / "cache"
        self.index = index
        self.client = FakeEmbeddingClient()
        self.indexer = DocumentIndexer(
            lambda: AsyncSession(
                bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
            ),
            cache_dir=self.cache_dir,
            # One attempt, so a failing client fails the version at once.
            embedder=Embedder(self.client, self.cache_dir, max_attempts=1, sleep=no_sleep),
            counter=WhitespaceTokenCounter(),
            index=index,
        )

    async def points(self) -> list[dict[str, Any]]:
        records, _ = await self.index.client.scroll(self.index.collection, limit=1000)
        return sorted((r.payload or {} for r in records), key=lambda p: p["chunk_id"])


@pytest.fixture
def admin(
    client: AsyncClient, connection: AsyncConnection, tmp_path: Path, policy_index: PolicyIndex
) -> Iterator[Admin]:
    # Depends on `client` so these overrides are set after, and removed
    # before, the client fixture's own.
    admin = Admin(connection, tmp_path, policy_index)
    app.dependency_overrides[get_document_indexer] = lambda: admin.indexer

    def document_service(db: DbSession) -> DocumentService:
        return DocumentService(db, upload_dir=admin.upload_dir)

    app.dependency_overrides[get_document_service] = document_service
    yield admin
    app.dependency_overrides.pop(get_document_indexer, None)
    app.dependency_overrides.pop(get_document_service, None)


@pytest.fixture
async def hr(client: AsyncClient, make_employee: MakeEmployee) -> EmployeeModel:
    employee = await make_employee(role=Role.HR)
    await login(client, employee)
    return employee


async def upload(
    client: AsyncClient,
    content: bytes = POLICY_V1,
    *,
    filename: str = "remote-work.md",
    **fields: Any,
) -> Response:
    form = {
        "slug": "remote-work-policy",
        "title": "Remote Work Policy",
        "access_level": "all",
        "version": "1",
        "effective_date": "2026-01-01",
    } | {key: str(value) for key, value in fields.items()}
    return await client.post(
        "/api/admin/documents", data=form, files={"file": (filename, content, "text/markdown")}
    )


async def documents(client: AsyncClient) -> dict[str, dict[str, Any]]:
    response = await client.get("/api/admin/documents")
    assert response.status_code == 200, response.text
    return {d["slug"]: d for d in response.json()}


# Upload


async def test_upload_indexes_the_document_in_the_background(
    client: AsyncClient, admin: Admin, hr: EmployeeModel, db: AsyncSession
) -> None:
    response = await upload(client)

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["slug"] == "remote-work-policy"
    assert body["version"]["version"] == 1
    # The response is sent before indexing starts.
    assert body["version"]["status"] == "indexing"

    document = (await documents(client))["remote-work-policy"]
    assert document["title"] == "Remote Work Policy"
    assert document["access_level"] == "all"
    assert document["active_version"] == 1
    [version] = document["versions"]
    assert version["status"] == "ready"
    assert version["chunk_count"] == 1
    assert version["effective_from"] == "2026-01-01"
    assert version["ingested_at"] is not None

    [point] = await admin.points()
    assert point["chunk_id"] == "remote-work-policy:1:0"
    assert "2 days a week" in point["text"]
    assert (admin.upload_dir / "remote-work-policy" / "v1.md").read_bytes() == POLICY_V1

    [row] = await audit_rows(db, AuditAction.DOCUMENT_UPLOAD)
    assert row.outcome is AuditOutcome.ALLOWED
    assert row.actor_id == hr.id
    assert row.target_id == uuid.UUID(body["version"]["id"])
    assert row.details["slug"] == "remote-work-policy"
    assert row.details["file_type"] == "markdown"


async def test_new_version_replaces_the_old_one(
    client: AsyncClient, admin: Admin, hr: EmployeeModel
) -> None:
    await upload(client)

    response = await upload(client, POLICY_V2, version=2)

    assert response.status_code == 202, response.text
    document = (await documents(client))["remote-work-policy"]
    assert document["active_version"] == 2
    assert [(v["version"], v["status"]) for v in document["versions"]] == [
        (2, "ready"),
        (1, "ready"),
    ]
    points = await admin.points()
    assert [p["chunk_id"] for p in points] == [
        "remote-work-policy:2:0",
        "remote-work-policy:2:1",
    ]
    assert "3 days a week" in points[0]["text"]


async def test_failed_version_keeps_the_old_one_active(
    client: AsyncClient, admin: Admin, hr: EmployeeModel
) -> None:
    await upload(client)

    await upload(client, BLANK, version=2)

    document = (await documents(client))["remote-work-policy"]
    assert document["active_version"] == 1
    assert [(v["version"], v["status"]) for v in document["versions"]] == [
        (2, "failed"),
        (1, "ready"),
    ]
    assert [p["chunk_id"] for p in await admin.points()] == ["remote-work-policy:1:0"]


async def test_stricter_access_level_applies_to_indexed_points_at_once(
    client: AsyncClient, admin: Admin, hr: EmployeeModel
) -> None:
    await upload(client)

    # v2 never indexes, but the document is HR-only from this request on.
    await upload(client, BLANK, version=2, access_level="hr", title="Remote Work (HR)")

    document = (await documents(client))["remote-work-policy"]
    assert document["access_level"] == "hr"
    assert document["active_version"] == 1
    [point] = await admin.points()
    assert point["access_level"] == "hr"
    assert point["title"] == "Remote Work (HR)"


async def test_docx_and_pdf_are_saved_with_their_own_extension(
    client: AsyncClient, admin: Admin, hr: EmployeeModel
) -> None:
    response = await upload(client, make_docx(), filename="../../etc/passwd.docx")

    assert response.status_code == 202, response.text
    # The client's file name never reaches the path.
    saved = list(admin.upload_dir.rglob("*"))
    assert [p.relative_to(admin.upload_dir).as_posix() for p in saved if p.is_file()] == [
        "remote-work-policy/v1.docx"
    ]
    assert (await documents(client))["remote-work-policy"]["versions"][0]["status"] == "ready"


# Refusals


async def test_version_must_be_newer_than_every_existing_one(
    client: AsyncClient, admin: Admin, hr: EmployeeModel, db: AsyncSession
) -> None:
    await upload(client, version=2)

    for version in (1, 2):
        response = await upload(client, POLICY_V2, version=version)
        assert response.status_code == 422
        assert response.json()["code"] == "version_conflict"
        assert response.json()["details"] == {"latest_version": 2}

    assert [p.name for p in (admin.upload_dir / "remote-work-policy").iterdir()] == ["v2.md"]
    denied = [
        r
        for r in await audit_rows(db, AuditAction.DOCUMENT_UPLOAD)
        if r.outcome is AuditOutcome.DENIED
    ]
    assert [r.details["reason"] for r in denied] == ["version_conflict", "version_conflict"]


@pytest.mark.parametrize(
    ("filename", "content", "status", "code"),
    [
        ("policy.pdf", POLICY_V1, 422, "invalid_upload"),
        ("policy.exe", b"MZ\x90\x00", 422, "invalid_upload"),
        ("policy.md", b"", 422, "invalid_upload"),
        ("policy.md", b"#" * (MAX_UPLOAD_BYTES + 1), 413, "file_too_large"),
    ],
)
async def test_invalid_files_are_refused_and_audited(
    client: AsyncClient,
    admin: Admin,
    hr: EmployeeModel,
    db: AsyncSession,
    filename: str,
    content: bytes,
    status: int,
    code: str,
) -> None:
    response = await upload(client, content, filename=filename)

    assert response.status_code == status
    assert response.json()["code"] == code
    assert not admin.upload_dir.exists()
    assert "remote-work-policy" not in await documents(client)
    [row] = await audit_rows(db, AuditAction.DOCUMENT_UPLOAD)
    assert row.outcome is AuditOutcome.DENIED
    assert row.details["reason"] == code


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("slug", "Remote Work"),
        ("slug", "../escape"),
        ("access_level", "public"),
        ("version", "0"),
        ("title", ""),
    ],
)
async def test_invalid_form_fields_are_refused(
    client: AsyncClient, admin: Admin, hr: EmployeeModel, field: str, value: str
) -> None:
    response = await upload(client, **{field: value})

    assert response.status_code == 422
    assert not admin.upload_dir.exists()


@pytest.mark.parametrize("role", [Role.EMPLOYEE, Role.MANAGER])
async def test_only_hr_can_use_the_document_endpoints(
    client: AsyncClient, admin: Admin, make_employee: MakeEmployee, role: Role
) -> None:
    await login(client, await make_employee(role=role))
    document_id = "00000000-0000-0000-0000-000000000000"

    assert (await upload(client)).status_code == 403
    assert (await client.get("/api/admin/documents")).status_code == 403
    assert (await client.post(f"/api/admin/documents/{document_id}/reindex")).status_code == 403
    assert not admin.upload_dir.exists()
    assert await admin.points() == []


async def test_upload_without_an_indexer_is_unavailable(
    client: AsyncClient, admin: Admin, hr: EmployeeModel
) -> None:
    app.dependency_overrides.pop(get_document_indexer)
    app.dependency_overrides[get_agent] = lambda: AgentResources(
        checkpointer=InMemorySaver(), chat_model=None, retriever=None, indexer=None
    )
    try:
        response = await upload(client)
    finally:
        app.dependency_overrides.pop(get_agent)

    assert response.status_code == 503
    assert response.json()["code"] == "unavailable"
    assert not admin.upload_dir.exists()


# Reindex


async def test_reindex_retries_a_failed_version(
    client: AsyncClient, admin: Admin, hr: EmployeeModel, db: AsyncSession
) -> None:
    admin.client.fail_times = 1  # the embedding API is down for the upload
    await upload(client)
    document = (await documents(client))["remote-work-policy"]
    assert document["versions"][0]["status"] == "failed"
    assert document["active_version"] is None

    response = await client.post(f"/api/admin/documents/{document['id']}/reindex")

    assert response.status_code == 202, response.text
    assert response.json()["version"]["status"] == "indexing"
    document = (await documents(client))["remote-work-policy"]
    assert document["versions"][0]["status"] == "ready"
    assert document["active_version"] == 1
    assert len(await admin.points()) == 1
    [row] = await audit_rows(db, AuditAction.DOCUMENT_REINDEX)
    assert row.outcome is AuditOutcome.ALLOWED
    assert row.details == row.details | {
        "slug": "remote-work-policy",
        "version": 1,
        "previous_status": "failed",
    }


async def test_reindex_unknown_document_is_not_found(
    client: AsyncClient, admin: Admin, hr: EmployeeModel
) -> None:
    response = await client.post(
        "/api/admin/documents/00000000-0000-0000-0000-000000000000/reindex"
    )

    assert response.status_code == 404
