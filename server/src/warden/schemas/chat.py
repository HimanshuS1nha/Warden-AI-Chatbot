import uuid
from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, StringConstraints

from warden.models import MessageRole
from warden.services.chat import MAX_MESSAGE_LENGTH


class ThreadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    # Null until the first message.
    title: str | None
    created_at: datetime


class ThreadPage(BaseModel):
    threads: list[ThreadResponse]
    # Pass as `before` for the next page; null on the last one.
    next: uuid.UUID | None


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: MessageRole
    content: str
    # On assistant replies: [{n, doc_slug, title, version, section, page, text}].
    citations: list[dict[str, Any]]
    pending_action_id: uuid.UUID | None
    created_at: datetime


class MessagePage(BaseModel):
    messages: list[MessageResponse]
    next: uuid.UUID | None


class SendMessage(BaseModel):
    # Whitespace is trimmed first, so "   " is empty and refused.
    message: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_LENGTH)
    ]
