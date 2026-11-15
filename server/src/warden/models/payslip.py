import uuid
from decimal import Decimal

from sqlalchemy import CheckConstraint, ForeignKey, Numeric, SmallInteger, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from warden.core.database import Base, UUIDPrimaryKey


class PayslipModel(UUIDPrimaryKey, Base):
    __tablename__ = "payslips"
    __table_args__ = (
        UniqueConstraint("employee_id", "year", "month"),
        CheckConstraint("month BETWEEN 1 AND 12", name="valid_month"),
        CheckConstraint("net_pay >= 0", name="net_pay_non_negative"),
    )

    # No separate index: the unique constraint's index starts with employee_id.
    employee_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("employees.id", ondelete="CASCADE"))
    month: Mapped[int] = mapped_column(SmallInteger)
    year: Mapped[int] = mapped_column(SmallInteger)
    net_pay: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    # Where the payslip PDF lives, if one was generated.
    document_ref: Mapped[str | None] = mapped_column(String(500))
