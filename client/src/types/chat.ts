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

// Mirror MessageResponse in server/src/warden/schemas/chat.py.
export type Message = {
  id: string;
  role: MessageRole;
  content: string;
  citations: Citation[];
  pending_action_id: string | null;
  created_at: string;
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

export type Feedback = "up" | "down";

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
  status: "streaming" | "done" | "error";
  // From the `error` event.
  error: string | null;
  tools: ToolCall[];
  citations: Citation[];
  action: PendingAction | null;
  feedback: Feedback | null;
};

export type ChatMessage = UserChatMessage | AssistantChatMessage;
