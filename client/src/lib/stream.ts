import { refreshSession } from "@/lib/axios";

import type { StreamEvent } from "@/types/chat";

// A non-2xx response before the stream started. The server's error body has
// `message` and, for service errors, a stable `code` (e.g. action_expired).
export class StreamHttpError extends Error {
  readonly status: number;
  readonly code: string | null;

  constructor(status: number, message: string, code: string | null) {
    super(message);
    this.name = "StreamHttpError";
    this.status = status;
    this.code = code;
  }
}

async function toHttpError(response: Response): Promise<StreamHttpError> {
  try {
    const body = (await response.json()) as { message?: string; code?: string };
    return new StreamHttpError(
      response.status,
      body.message ?? response.statusText,
      body.code ?? null,
    );
  } catch {
    return new StreamHttpError(response.status, response.statusText, null);
  }
}

// One SSE block: "event: name" and "data: json" lines.
function parseEvent(block: string): StreamEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (data.length === 0) return null;
  try {
    return { event, data: JSON.parse(data.join("\n")) } as StreamEvent;
  } catch {
    return null;
  }
}

/**
 * POST to a streaming chat endpoint and call `onEvent` for each server-sent
 * event. Resolves when the stream ends.
 *
 * Uses fetch, not axios, to read the body as it arrives, so it handles the
 * expired-access-token case itself: one refresh, then one retry.
 *
 * Throws StreamHttpError for a non-2xx response, a DOMException named
 * AbortError when `signal` aborts, and a TypeError for network failures.
 */
export async function streamChat(
  path: string,
  body: unknown,
  onEvent: (event: StreamEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const post = () =>
    fetch(`${import.meta.env.VITE_API_URL}${path}`, {
      method: "POST",
      credentials: "include",
      headers: {
        Accept: "text/event-stream",
        ...(body !== undefined && { "Content-Type": "application/json" }),
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal,
    });

  let response = await post();
  if (response.status === 401) {
    try {
      await refreshSession();
    } catch {
      throw new StreamHttpError(401, "Your session has expired. Please log in again.", null);
    }
    response = await post();
  }
  if (!response.ok) throw await toHttpError(response);
  if (!response.body) throw new TypeError("The response has no body.");

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    // Chunks must be read in order, one at a time.
    // oxlint-disable-next-line no-await-in-loop
    const { done, value } = await reader.read();
    if (done) break;
    // Normalise line endings over the whole buffer so a \r\n split across
    // chunks still becomes one \n.
    buffer = (buffer + value).replaceAll("\r\n", "\n");
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const event = parseEvent(buffer.slice(0, boundary));
      buffer = buffer.slice(boundary + 2);
      if (event) onEvent(event);
      boundary = buffer.indexOf("\n\n");
    }
  }
}
