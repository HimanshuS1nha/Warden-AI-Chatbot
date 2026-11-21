import { axios } from "@/lib/axios";

import { API_ROUTES } from "@/constants/api-routes";

import type { Employee } from "@/types/employee";
import type { LoginValidatorType } from "@/validators/login.validator";

export async function login(credentials: LoginValidatorType): Promise<Employee> {
  const { data } = await axios.post<Employee>(API_ROUTES.auth.login, credentials);
  return data;
}
