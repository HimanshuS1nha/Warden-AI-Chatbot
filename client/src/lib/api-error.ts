import { isAxiosError } from "axios";

// Every server error body has a human-readable `message`
// (server/src/warden/core/exceptions.py).
export function getErrorMessage(error: unknown): string {
  if (isAxiosError<{ message?: string }>(error)) {
    if (!error.response) return "Can't reach the server. Check your connection and try again.";
    const message = error.response.data?.message;
    if (message) return message;
  }
  return "Something went wrong. Please try again.";
}
