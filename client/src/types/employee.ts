// Mirrors EmployeeResponse in server/src/warden/schemas/employee.py.
export type Role = "employee" | "manager" | "hr";

export type Employee = {
  id: string;
  employee_code: string;
  name: string;
  email: string;
  role: Role;
  department_id: string;
  manager_id: string | null;
  job_title: string;
  joining_date: string;
};
