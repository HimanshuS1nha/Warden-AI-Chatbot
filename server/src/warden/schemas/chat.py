import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

from warden.models import MessageRole, PendingActionStatus, PendingActionType
from warden.services.chat import MAX_FEEDBACK_COMMENT_LENGTH, MAX_MESSAGE_LENGTH, TITLE_LENGTH

# 1 = thumbs up, -1 = thumbs down.
Rating = Literal[1, -1]


class ThreadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    # Null until the first message.
    title: str | None
    created_at: datetime


class RenameThread(BaseModel):
    # Trimmed first, so a blank title is refused.
    title: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=TITLE_LENGTH)
    ]


class ThreadPage(BaseModel):
    threads: list[ThreadResponse]
    # Pass as `before` for the next page; null on the last one.
    next: uuid.UUID | None


class MessageAction(BaseModel):
    """The change a reply proposed, for its confirmation card."""

    id: uuid.UUID
    action_type: PendingActionType
    # E.g. "Casual leave, Mon 17 Nov to Tue 18 Nov, 2 days. Balance after: 6".
    summary: str
    # As it is now. A pending action past `expires_at` is reported expired.
    status: PendingActionStatus
    expires_at: datetime


class MessageResponse(BaseModel):
    id: uuid.UUID
    role: MessageRole
    content: str
    # On assistant replies: [{n, doc_slug, title, version, section, page, text}].
    citations: list[dict[str, Any]]
    # On a reply that proposed a change; null otherwise.
    action: MessageAction | None
    # The caller's rating of this reply, if they gave one.
    feedback: Rating | None
    created_at: datetime


class MessagePage(BaseModel):
    messages: list[MessageResponse]
    next: uuid.UUID | None


class SendMessage(BaseModel):
    # Whitespace is trimmed first, so "   " is empty and refused.
    message: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_LENGTH)
    ]


class FeedbackRequest(BaseModel):
    message_id: uuid.UUID
    rating: Rating
    # Optional. Blank is the same as none.
    comment: (
        Annotated[
            str, StringConstraints(strip_whitespace=True, max_length=MAX_FEEDBACK_COMMENT_LENGTH)
        ]
        | None
    ) = None


class FeedbackResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    message_id: uuid.UUID
    rating: Rating
    comment: str | None
