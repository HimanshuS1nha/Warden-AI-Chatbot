from typing import TYPE_CHECKING

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from warden.core.database import Base

if TYPE_CHECKING:
    from warden.models.employee import EmployeeModel


class DepartmentModel(Base):
    __tablename__ = "departments"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)

    employees: Mapped[list["EmployeeModel"]] = relationship(
        back_populates="department", lazy="raise"
    )
