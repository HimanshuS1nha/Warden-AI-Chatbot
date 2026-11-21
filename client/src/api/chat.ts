import { infiniteQueryOptions } from "@tanstack/react-query";

import { axios } from "@/lib/axios";
import { isClientError } from "@/lib/api-error";

import { API_ROUTES } from "@/constants/api-routes";
import { QUERY_KEYS } from "@/constants/query-keys";

import type {
  ChatMessage,
  ChatMessagePage,
  Message,
  MessagePage,
  Thread,
  ThreadPage,
} from "@/types/chat";

const THREADS_PAGE_SIZE = 30;
const MESSAGES_PAGE_SIZE = 30;

export async function createThread(): Promise<Thread> {
  const { data } = await axios.post<Thread>(API_ROUTES.chat.threads);
  return data;
}

export async function getThreads(before: string | null): Promise<ThreadPage> {
  const { data } = await axios.get<ThreadPage>(API_ROUTES.chat.threads, {
    params: { limit: THREADS_PAGE_SIZE, ...(before && { before }) },
  });
  return data;
}

// The caller's threads, newest first, one cursor page at a time.
export const threadsInfiniteQueryOptions = infiniteQueryOptions({
  queryKey: QUERY_KEYS.threads,
  queryFn: ({ pageParam }) => getThreads(pageParam),
  initialPageParam: null as string | null,
  getNextPageParam: (lastPage) => lastPage.next,
});

// A saved message as the chat page shows it. Tool activity and confirmation
// cards only come from the live stream, so saved messages have none.
export function toChatMessage(message: Message): ChatMessage {
  if (message.role === "user") {
    return {
      id: message.id,
      role: "user",
      content: message.content,
      createdAt: message.created_at,
    };
  }
  return {
    id: message.id,
    role: "assistant",
    content: message.content,
    createdAt: message.created_at,
    status: "done",
    error: null,
    tools: [],
    citations: message.citations,
    action: null,
  };
}

export async function getMessages(
  threadId: string,
  before: string | null,
): Promise<ChatMessagePage> {
  const { data } = await axios.get<MessagePage>(API_ROUTES.chat.messages(threadId), {
    params: { limit: MESSAGES_PAGE_SIZE, ...(before && { before }) },
  });
  return { messages: data.messages.map(toChatMessage), next: data.next };
}

// A thread's messages, newest first, one cursor page at a time. Replies sent
// from this page are written into this cache by useChatStream, with the tool
// activity and confirmation cards the server doesn't store, so it's never
// refetched in the background: that would replace them with the bare saved
// messages.
export const messagesInfiniteQueryOptions = (threadId: string) =>
  infiniteQueryOptions({
    queryKey: QUERY_KEYS.messages(threadId),
    queryFn: ({ pageParam }) => getMessages(threadId, pageParam),
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage) => lastPage.next,
    staleTime: Infinity,
    // A 404 (unknown or someone else's thread) won't fix itself.
    retry: (failureCount, error) => !isClientError(error) && failureCount < 2,
  });
