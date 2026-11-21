export const API_ROUTES = {
  auth: {
    login: "/auth/login",
    refresh: "/auth/refresh",
    logout: "/auth/logout",
    me: "/auth/me",
  },
  chat: {
    threads: "/chat/threads",
    messages: (threadId: string) => `/chat/threads/${encodeURIComponent(threadId)}/messages`,
    confirmAction: (actionId: string) => `/chat/actions/${encodeURIComponent(actionId)}/confirm`,
    feedback: "/chat/feedback",
    feedbackFor: (messageId: string) => `/chat/feedback/${encodeURIComponent(messageId)}`,
    cancelAction: (actionId: string) => `/chat/actions/${encodeURIComponent(actionId)}/cancel`,
  },
} as const;
