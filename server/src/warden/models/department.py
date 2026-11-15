from typing import TYPE_CHECKING

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from warden.core.database import Base, UUIDPrimaryKey

if TYPE_CHECKING:
    from warden.models.employee import EmployeeModel


class DepartmentModel(UUIDPrimaryKey, Base):
    __tablename__ = "departments"

    name: Mapped[str] = mapped_column(String(100), unique=True)

    employees: Mapped[list["EmployeeModel"]] = relationship(
        back_populates="department", lazy="raise"
    )
