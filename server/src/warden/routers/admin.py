"""HR administration: policy documents. Used through /docs or curl; the web
client has no admin screens.

    curl -b cookies.txt -F file=@leave-policy-v3.pdf -F slug=leave-policy \
      -F "title=Leave Policy" -F access_level=all -F version=3 \
      http://localhost:8000/api/admin/documents
"""

import uuid
from datetime import date
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile, status

from warden.core.dependencies import CurrentUser, require_role
from warden.models import AccessLevel, Role
from warden.rag.indexer import DocumentIndexer
from warden.routers.chat import AgentDep
from warden.routers.services import DocumentServiceDep
from warden.schemas.document import DocumentResponse, IndexingStartedResponse
from warden.services.errors import Unavailable
from warden.services.leave import company_today
from warden.services.uploads import MAX_UPLOAD_BYTES

router = APIRouter(prefix="/api/admin", tags=["admin"])

HRUser = Annotated[CurrentUser, Depends(require_role(Role.HR))]

SLUG_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"


def get_document_indexer(agent: AgentDep) -> DocumentIndexer:
    if agent.indexer is None:
        raise Unavailable("Document indexing is off: the server has no embedding API key.")
    return agent.indexer


IndexerDep = Annotated[DocumentIndexer, Depends(get_document_indexer)]


@router.get("/documents")
async def list_documents(user: HRUser, documents: DocumentServiceDep) -> list[DocumentResponse]:
    """Every document with its versions, their status and chunk counts."""
    return [DocumentResponse.build(d) for d in await documents.list_documents(user)]


@router.post("/documents", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    user: HRUser,
    documents: DocumentServiceDep,
    indexer: IndexerDep,
    background: BackgroundTasks,
    file: Annotated[UploadFile, File(description="PDF, DOCX or Markdown, at most 10 MB")],
    slug: Annotated[str, Form(pattern=SLUG_PATTERN, max_length=100)],
    title: Annotated[str, Form(min_length=1, max_length=255)],
    access_level: Annotated[AccessLevel, Form()],
    version: Annotated[int, Form(gt=0)],
    effective_date: Annotated[date | None, Form()] = None,
) -> IndexingStartedResponse:
    """Add a document, or a new version of one, and index it in the
    background. It replaces the current version once indexing succeeds."""
    # One byte over the limit is enough to know it's too big.
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    document, row = await documents.add_version(
        user,
        slug=slug,
        title=title.strip(),
        access_level=access_level,
        version=version,
        effective_from=effective_date or company_today(),
        filename=file.filename,
        data=data,
        index=indexer,
    )
    background.add_task(indexer.run, row.id)
    return IndexingStartedResponse.build(document, row)


@router.post("/documents/{document_id}/reindex", status_code=status.HTTP_202_ACCEPTED)
async def reindex_document(
    document_id: uuid.UUID,
    user: HRUser,
    documents: DocumentServiceDep,
    indexer: IndexerDep,
    background: BackgroundTasks,
) -> IndexingStartedResponse:
    """Index the document's newest version again, in the background: retry a
    failed upload, or pick up parser and chunking changes."""
    document, row = await documents.start_reindex(user, document_id)
    background.add_task(indexer.run, row.id)
    return IndexingStartedResponse.build(document, row)
