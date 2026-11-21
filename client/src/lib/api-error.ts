import { isAxiosError } from "axios";

import { StreamHttpError } from "@/lib/stream";

const OFFLINE_MESSAGE = "Can't reach the server. Check your connection and try again.";

// Every server error body has a human-readable `message`
// (server/src/warden/core/exceptions.py).
export function getErrorMessage(error: unknown): string {
  if (isAxiosError<{ message?: string }>(error)) {
    if (!error.response) return OFFLINE_MESSAGE;
    const message = error.response.data?.message;
    if (message) return message;
  }
  if (error instanceof StreamHttpError) return error.message;
  // fetch rejects with a TypeError when the request never reaches the server.
  if (error instanceof TypeError) return OFFLINE_MESSAGE;
  return "Something went wrong. Please try again.";
}

// 4xx errors won't change on a retry.
export function isClientError(error: unknown): boolean {
  const status = isAxiosError(error) ? error.response?.status : undefined;
  return status !== undefined && status >= 400 && status < 500;
}
