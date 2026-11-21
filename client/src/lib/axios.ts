import { create, AxiosError, type InternalAxiosRequestConfig } from "axios";

import { API_ROUTES } from "@/constants/api-routes";

declare module "axios" {
  interface InternalAxiosRequestConfig {
    // Set on a request after it has been retried once following a refresh,
    // so a second 401 fails instead of looping.
    retried?: boolean;
  }
}

export const axios = create({
  baseURL: import.meta.env.VITE_API_URL,
  // Auth tokens live in httpOnly cookies; send them on cross-origin requests.
  withCredentials: true,
});

// A 401 from these routes means bad credentials or a dead session, not an
// expired access token, so refreshing would not help.
const NO_REFRESH_ROUTES: ReadonlySet<string> = new Set([
  API_ROUTES.auth.login,
  API_ROUTES.auth.refresh,
]);

// Shared by every request that gets a 401 while a refresh is in flight. The
// server rotates the refresh token, so two parallel refreshes would make the
// second one look like token reuse.
let refreshPromise: Promise<void> | null = null;

// Exported for the chat stream, which uses fetch and so bypasses this interceptor.
export function refreshSession(): Promise<void> {
  refreshPromise ??= axios
    .post(API_ROUTES.auth.refresh)
    .then(() => undefined)
    .finally(() => {
      refreshPromise = null;
    });
  return refreshPromise;
}

axios.interceptors.response.use(
  (response) => response,
  async (error: unknown) => {
    if (!(error instanceof AxiosError)) throw error;

    const config: InternalAxiosRequestConfig | undefined = error.config;
    if (
      error.response?.status !== 401 ||
      !config ||
      config.retried ||
      NO_REFRESH_ROUTES.has(config.url ?? "")
    ) {
      throw error;
    }

    config.retried = true;
    try {
      await refreshSession();
    } catch {
      // Refresh failed: the session is gone. Surface the original 401 so
      // callers (e.g. the root route's auth check) can send the user to login.
      throw error;
    }
    return axios(config);
  },
);
