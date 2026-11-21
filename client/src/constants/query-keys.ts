export const QUERY_KEYS = {
  me: ["auth", "me"],
  threads: ["chat", "threads"],
  messages: (threadId: string) => ["chat", "messages", threadId] as const,
} as const;
