"""Role checks inside services, audited the same way as `require_role`.

Routers already guard their routes with `require_role`. Services check again
because agent tools call them directly, without a route in between.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from warden.core.dependencies import CurrentUser
from warden.models import AuditAction, AuditOutcome, Role
from warden.services import audit
from warden.services.errors import Forbidden


async def ensure_role(db: AsyncSession, user: CurrentUser, *roles: Role, operation: str) -> None:
    """Raise Forbidden unless the user has one of `roles`. The refusal is
    recorded as `authz.forbidden` and committed before raising."""
    if user.role in roles:
        return
    audit.record(
        db,
        action=AuditAction.AUTHZ_FORBIDDEN,
        outcome=AuditOutcome.DENIED,
        actor_id=user.employee_id,
        details={
            "operation": operation,
            "role": user.role.value,
            "required": sorted(role.value for role in roles),
        },
    )
    await db.commit()
    raise Forbidden("You don't have permission to do this")
