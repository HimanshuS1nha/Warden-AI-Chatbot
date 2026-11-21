import type { InfiniteData, QueryClient } from "@tanstack/react-query";

import { QUERY_KEYS } from "@/constants/query-keys";

import type { AssistantChatMessage, ChatMessagePage } from "@/types/chat";

// A thread's messages in the query cache: pages newest first, and each page's
// messages newest first, as the API returns them.
export type MessagesData = InfiniteData<ChatMessagePage, string | null>;

export const EMPTY_MESSAGES: MessagesData = {
  pages: [{ messages: [], next: null }],
  pageParams: [null],
};

export function updateMessagePages(
  queryClient: QueryClient,
  threadId: string,
  fn: (pages: ChatMessagePage[]) => ChatMessagePage[],
) {
  queryClient.setQueryData<MessagesData>(
    QUERY_KEYS.messages(threadId),
    (data = EMPTY_MESSAGES) => ({
      ...data,
      pages: fn(data.pages),
    }),
  );
}

export function updateAssistantMessage(
  queryClient: QueryClient,
  threadId: string,
  messageId: string,
  fn: (message: AssistantChatMessage) => AssistantChatMessage,
) {
  updateMessagePages(queryClient, threadId, (pages) =>
    pages.map((page) => ({
      ...page,
      messages: page.messages.map((m) =>
        m.id === messageId && m.role === "assistant" ? fn(m) : m,
      ),
    })),
  );
}
