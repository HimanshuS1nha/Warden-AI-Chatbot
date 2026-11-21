import { queryOptions } from "@tanstack/react-query";

import { axios } from "@/lib/axios";

import { API_ROUTES } from "@/constants/api-routes";
import { QUERY_KEYS } from "@/constants/query-keys";

import type { Employee } from "@/types/employee";
import type { LoginValidatorType } from "@/validators/login.validator";

export async function login(credentials: LoginValidatorType): Promise<Employee> {
  const { data } = await axios.post<Employee>(API_ROUTES.auth.login, credentials);
  return data;
}

export async function logout(): Promise<void> {
  await axios.post(API_ROUTES.auth.logout);
}

export async function getMe(): Promise<Employee> {
  const { data } = await axios.get<Employee>(API_ROUTES.auth.me);
  return data;
}

export const meQueryOptions = queryOptions({
  queryKey: QUERY_KEYS.me,
  queryFn: getMe,
  // The user only changes on login and logout, and both update this cache
  // directly, so it never needs a background refetch.
  staleTime: Infinity,
  // A failure means there's no session (the axios interceptor has already
  // tried a refresh). Retrying would only delay the redirect to /login.
  retry: false,
});
