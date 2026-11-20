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
    # Confirmed, but the service refused the write when it ran.
    FAILED = "failed"


class PendingActionType(StrEnum):
    """The write a pending action runs on confirm. Stored as a plain string,
    so adding a type needs no migration."""

    APPLY_LEAVE = "apply_leave"
    CANCEL_LEAVE = "cancel_leave"
    APPROVE_LEAVE = "approve_leave"
    REJECT_LEAVE = "reject_leave"


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


class AuditAction(StrEnum):
    """What an audit log row records. Stored as a plain string, so adding an
    action needs no migration."""

    AUTH_LOGIN = "auth.login"
    AUTH_LOGOUT = "auth.logout"
    AUTH_REFRESH_REUSE = "auth.refresh_reuse"
    AUTHZ_FORBIDDEN = "authz.forbidden"
    LEAVE_APPLY = "leave.apply"
    LEAVE_CANCEL = "leave.cancel"
    LEAVE_APPROVE = "leave.approve"
    LEAVE_REJECT = "leave.reject"
    EMPLOYEE_VIEW = "employee.view"
    PENDING_ACTION_CONFIRM = "pending_action.confirm"
    PENDING_ACTION_CANCEL = "pending_action.cancel"
    DOCUMENT_UPLOAD = "document.upload"
    DOCUMENT_REINDEX = "document.reindex"


class AuditTarget(StrEnum):
    """The kind of record an audit log row's `target_id` points at."""

    EMPLOYEE = "employee"
    SESSION = "session"
    LEAVE_REQUEST = "leave_request"
    PENDING_ACTION = "pending_action"
    DOCUMENT_VERSION = "document_version"
