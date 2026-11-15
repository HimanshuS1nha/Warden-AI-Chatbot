from enum import StrEnum

from sqlalchemy import Enum


def pg_enum(enum_cls: type[StrEnum], name: str) -> Enum:
    """Postgres enum type that stores the lowercase values ("employee"), not
    the member names ("EMPLOYEE")."""
    return Enum(enum_cls, name=name, values_callable=lambda e: [m.value for m in e])


class Role(StrEnum):
    EMPLOYEE = "employee"
    MANAGER = "manager"
    HR = "hr"


class LeaveType(StrEnum):
    CASUAL = "casual"
    SICK = "sick"
    EARNED = "earned"


class LeaveStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class AttendanceStatus(StrEnum):
    PRESENT = "present"
    ABSENT = "absent"
    WFH = "wfh"
    ON_LEAVE = "on_leave"
    HOLIDAY = "holiday"


class AccessLevel(StrEnum):
    """Who can retrieve a document. Employees see ALL, managers see ALL and
    MANAGER, HR sees everything."""

    ALL = "all"
    MANAGER = "manager"
    HR = "hr"


class DocumentVersionStatus(StrEnum):
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class PendingActionStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class AuditOutcome(StrEnum):
    ALLOWED = "allowed"
    DENIED = "denied"
    ERROR = "error"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class UnansweredReason(StrEnum):
    NO_RELEVANT_DOCS = "no_relevant_docs"
    LOW_CONFIDENCE = "low_confidence"
