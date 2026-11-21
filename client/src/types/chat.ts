// Mirror ThreadResponse and ThreadPage in server/src/warden/schemas/chat.py.
export type Thread = {
  id: string;
  // Null until the first message is sent.
  title: string | null;
  created_at: string;
};

export type ThreadPage = {
  threads: Thread[];
  // Pass as `before` to get the next page; null on the last one.
  next: string | null;
};

// Mirror Citation fields in server/src/warden/rag/citations.py (CITATION_FIELDS).
export type Citation = {
  // The marker number used in the reply text, e.g. [1].
  n: number;
  doc_slug: string;
  title: string;
  version: number;
  section: string;
  page: number | null;
  text: string;
};

// Mirror MessageRole, PendingActionType and PendingActionStatus in
// server/src/warden/models/enums.py.
export type MessageRole = "user" | "assistant";
export type PendingActionType = "apply_leave" | "cancel_leave" | "approve_leave" | "reject_leave";
export type PendingActionStatus = "pending" | "confirmed" | "cancelled" | "expired" | "failed";

// Mirror MessageResponse and MessagePage in server/src/warden/schemas/chat.py.
export type Message = {
  id: string;
  role: MessageRole;
  content: string;
  citations: Citation[];
  pending_action_id: string | null;
  created_at: string;
};

export type MessagePage = {
  messages: Message[];
  // Pass as `before` to get older messages; null on the last page.
  next: string | null;
};

// What the chat page renders: a saved message plus what the stream adds
// (tool activity, the confirmation card, streaming and error states).
export type ToolCall = {
  id: string;
  name: string;
  status: "running" | "ok" | "failed";
};

export type PendingAction = {
  id: string;
  type: PendingActionType;
  summary: string;
  status: PendingActionStatus;
};

export type UserChatMessage = {
  id: string;
  role: "user";
  content: string;
  createdAt: string;
};

export type AssistantChatMessage = {
  id: string;
  role: "assistant";
  // Joined tokens while streaming, then the saved reply from `done`.
  content: string;
  createdAt: string;
  // "stopped": the user pressed Stop or left the page mid-reply.
  status: "streaming" | "done" | "error" | "stopped";
  // From the `error` event, or why the request failed.
  error: string | null;
  tools: ToolCall[];
  citations: Citation[];
  action: PendingAction | null;
};

export type ChatMessage = UserChatMessage | AssistantChatMessage;

// A page of the thread's messages in the query cache, newest first like the API.
export type ChatMessagePage = {
  messages: ChatMessage[];
  next: string | null;
};

// Server-sent events from the message, confirm and cancel endpoints
// (server/src/warden/graph/streaming.py).
export type StreamEvent =
  | {
      event: "action_result";
      data: { action_id: string; status: PendingActionStatus; ok: boolean };
    }
  | { event: "token"; data: { text: string } }
  | { event: "tool_start"; data: { id: string; name: string; args: Record<string, unknown> } }
  | { event: "tool_end"; data: { id: string; name: string; ok: boolean } }
  | { event: "citations"; data: { citations: Citation[] } }
  | {
      event: "confirmation_required";
      data: { action_id: string; action_type: PendingActionType; summary: string };
    }
  | { event: "done"; data: { message_id: string; content: string } }
  | { event: "error"; data: { message: string } };
