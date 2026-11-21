"""One service per request, built on the request's database session.

Routers depend on these instead of building services inline, so tests can
override one, e.g. to give LeaveService a fixed date.
"""

from typing import Annotated

from fastapi import Depends

from warden.core.config import get_settings
from warden.core.dependencies import DbSession
from warden.services.admin import AdminService
from warden.services.attendance import AttendanceService
from warden.services.documents import DocumentService
from warden.services.employee import EmployeeService
from warden.services.holiday import HolidayService
from warden.services.leave import LeaveService
from warden.services.payslip import PayslipService


def get_leave_service(db: DbSession) -> LeaveService:
    return LeaveService(db)


def get_employee_service(db: DbSession) -> EmployeeService:
    return EmployeeService(db)


def get_attendance_service(db: DbSession) -> AttendanceService:
    return AttendanceService(db)


def get_payslip_service(db: DbSession) -> PayslipService:
    return PayslipService(db)


def get_holiday_service(db: DbSession) -> HolidayService:
    return HolidayService(db)


def get_admin_service(db: DbSession) -> AdminService:
    return AdminService(db)


def get_document_service(db: DbSession) -> DocumentService:
    return DocumentService(db, upload_dir=get_settings().upload_dir)


LeaveServiceDep = Annotated[LeaveService, Depends(get_leave_service)]
EmployeeServiceDep = Annotated[EmployeeService, Depends(get_employee_service)]
AttendanceServiceDep = Annotated[AttendanceService, Depends(get_attendance_service)]
PayslipServiceDep = Annotated[PayslipService, Depends(get_payslip_service)]
HolidayServiceDep = Annotated[HolidayService, Depends(get_holiday_service)]
AdminServiceDep = Annotated[AdminService, Depends(get_admin_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
