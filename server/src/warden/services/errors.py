"""Errors services raise. Routers and agent tools turn them into responses.

- NotFound: the record doesn't exist, or exists outside the caller's scope.
  The two are deliberately the same error, so probing reveals nothing.
- Forbidden: the caller's role can't use this operation at all.
- RuleViolation: a business rule refused the request. Each subclass has a
  stable `code` that clients and the agent can rely on.
"""

from typing import Any, ClassVar


class ServiceError(Exception):
    code: ClassVar[str] = "service_error"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        # Extra facts for the client or the agent, e.g. the available balance.
        self.details = details


class NotFound(ServiceError):
    code = "not_found"


class Forbidden(ServiceError):
    code = "forbidden"


class RuleViolation(ServiceError):
    code = "rule_violation"


class InvalidInput(RuleViolation):
    code = "invalid_input"


class InvalidDates(RuleViolation):
    code = "invalid_dates"


class BackdateNotAllowed(RuleViolation):
    code = "backdate_not_allowed"


class Overlap(RuleViolation):
    code = "overlap"


class InsufficientBalance(RuleViolation):
    code = "insufficient_balance"


class InvalidStatus(RuleViolation):
    """The request isn't in a status that allows this, e.g. approving one
    that was already rejected, or cancelling leave that has started."""

    code = "invalid_status"
