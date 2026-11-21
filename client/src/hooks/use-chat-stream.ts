import { type InfiniteData, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import { getErrorMessage } from "@/lib/api-error";
import { type MessagesData, updateAssistantMessage, updateMessagePages } from "@/lib/chat-cache";
import { StreamHttpError, streamChat } from "@/lib/stream";

import { API_ROUTES } from "@/constants/api-routes";
import { QUERY_KEYS } from "@/constants/query-keys";

import type {
  AssistantChatMessage,
  ChatMessage,
  ChatMessagePage,
  PendingActionStatus,
  StreamEvent,
  ThreadPage,
  UserChatMessage,
} from "@/types/chat";

// Ids for messages the server hasn't confirmed yet. The reply gets its real
// id from `done`; the user's message keeps this one until the next refetch.
const localId = () => `local-${crypto.randomUUID()}`;

function newAssistantMessage(): AssistantChatMessage {
  return {
    id: localId(),
    role: "assistant",
    content: "",
    createdAt: new Date().toISOString(),
    status: "streaming",
    error: null,
    tools: [],
    citations: [],
    action: null,
    feedback: null,
  };
}

/**
 * Sends messages and confirms or cancels proposed actions in one thread,
 * streaming each reply into the thread's messages cache. One stream at a
 * time; leaving the page stops it.
 */
export function useChatStream(threadId: string) {
  const queryClient = useQueryClient();
  const messagesKey = QUERY_KEYS.messages(threadId);
  const abortRef = useRef<AbortController | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  // The action whose Confirm or Cancel is in flight, for the card's spinner.
  const [busyAction, setBusyAction] = useState<{
    id: string;
    decision: "confirm" | "cancel";
  } | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  // Cache helpers. Page 0 holds the newest messages, newest first.

  const updatePages = (fn: (pages: ChatMessagePage[]) => ChatMessagePage[]) =>
    updateMessagePages(queryClient, threadId, fn);

  // `messages` oldest first, as they happen.
  const append = (...messages: ChatMessage[]) => {
    updatePages(([first = { messages: [], next: null }, ...rest]) => [
      { ...first, messages: [...messages.toReversed(), ...first.messages] },
      ...rest,
    ]);
  };

  const remove = (...ids: string[]) => {
    updatePages((pages) =>
      pages.map((page) => ({
        ...page,
        messages: page.messages.filter((m) => !ids.includes(m.id)),
      })),
    );
  };

  const updateAssistant = (
    id: string,
    fn: (message: AssistantChatMessage) => AssistantChatMessage,
  ) => updateAssistantMessage(queryClient, threadId, id, fn);

  const setActionStatus = (actionId: string, status: PendingActionStatus) => {
    updatePages((pages) =>
      pages.map((page) => ({
        ...page,
        messages: page.messages.map((m) =>
          m.role === "assistant" && m.action?.id === actionId
            ? { ...m, action: { ...m.action, status } }
            : m,
        ),
      })),
    );
  };

  // Applies one stream event to the reply being written. Returns true for the
  // events that end the turn.
  const applyEvent = (replyId: string, { event, data }: StreamEvent): boolean => {
    switch (event) {
      case "action_result":
        setActionStatus(data.action_id, data.status);
        return false;
      case "token":
        updateAssistant(replyId, (m) => ({ ...m, content: m.content + data.text }));
        return false;
      case "tool_start":
        updateAssistant(replyId, (m) => ({
          ...m,
          tools: [...m.tools, { id: data.id, name: data.name, status: "running" }],
        }));
        return false;
      case "tool_end":
        updateAssistant(replyId, (m) => ({
          ...m,
          tools: m.tools.map((t) =>
            t.id === data.id ? { ...t, status: data.ok ? "ok" : "failed" } : t,
          ),
        }));
        return false;
      case "citations":
        updateAssistant(replyId, (m) => ({ ...m, citations: data.citations }));
        return false;
      case "confirmation_required":
        updateAssistant(replyId, (m) => ({
          ...m,
          action: {
            id: data.action_id,
            type: data.action_type,
            summary: data.summary,
            status: "pending",
          },
        }));
        return false;
      case "done":
        // The saved reply, with invented citation markers removed. It must
        // be the last update keyed by replyId, since the id changes here.
        updateAssistant(replyId, (m) => ({
          ...m,
          id: data.message_id,
          content: data.content,
          status: "done",
        }));
        return true;
      case "error":
        updateAssistant(replyId, (m) => ({ ...m, status: "error", error: data.message }));
        return true;
      default:
        return false;
    }
  };

  /**
   * Streams one reply into the message `replyId`. A refusal before the
   * stream starts goes to `handleHttpError` when given; every other failure
   * is shown on the reply.
   */
  const runStream = async (
    path: string,
    body: unknown,
    replyId: string,
    handleHttpError?: (error: StreamHttpError) => void,
  ) => {
    const controller = new AbortController();
    abortRef.current = controller;
    setIsStreaming(true);
    let ended = false;
    try {
      await streamChat(
        path,
        body,
        (event) => {
          if (applyEvent(replyId, event)) ended = true;
        },
        controller.signal,
      );
      if (!ended) {
        updateAssistant(replyId, (m) => ({
          ...m,
          status: "error",
          error: "The reply was cut off. Please try again.",
        }));
      }
    } catch (error) {
      if (controller.signal.aborted) {
        updateAssistant(replyId, (m) => ({ ...m, status: "stopped" }));
      } else if (error instanceof StreamHttpError && handleHttpError) {
        handleHttpError(error);
      } else {
        updateAssistant(replyId, (m) => ({
          ...m,
          status: "error",
          error: getErrorMessage(error),
        }));
      }
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setIsStreaming(false);
    }
  };

  // The first message names the thread on the server; refetch the list then.
  const refreshThreadTitle = () => {
    const threads = queryClient.getQueryData<InfiniteData<ThreadPage>>(QUERY_KEYS.threads);
    const thread = threads?.pages.flatMap((p) => p.threads).find((t) => t.id === threadId);
    if (!thread || thread.title === null) {
      void queryClient.invalidateQueries({ queryKey: QUERY_KEYS.threads });
    }
  };

  const send = async (text: string) => {
    const content = text.trim();
    if (!content || isStreaming) return;
    const question: UserChatMessage = {
      id: localId(),
      role: "user",
      content,
      createdAt: new Date().toISOString(),
    };
    const reply = newAssistantMessage();
    append(question, reply);
    await runStream(API_ROUTES.chat.messages(threadId), { message: content }, reply.id);
    refreshThreadTitle();
  };

  // Drops the failed reply and the question before it, then asks again.
  const retry = async (failedReplyId: string) => {
    if (isStreaming) return;
    const data = queryClient.getQueryData<MessagesData>(messagesKey);
    const newestFirst = data?.pages.flatMap((p) => p.messages) ?? [];
    const index = newestFirst.findIndex((m) => m.id === failedReplyId);
    const question = newestFirst[index + 1];
    if (index === -1 || question?.role !== "user") return;
    remove(failedReplyId, question.id);
    await send(question.content);
  };

  const decide = async (actionId: string, decision: "confirm" | "cancel") => {
    if (isStreaming) return;
    setBusyAction({ id: actionId, decision });
    const reply = newAssistantMessage();
    append(reply);
    const path =
      decision === "confirm"
        ? API_ROUTES.chat.confirmAction(actionId)
        : API_ROUTES.chat.cancelAction(actionId);
    await runStream(path, undefined, reply.id, (error) => {
      // Refused before anything ran (already handled, expired, not found):
      // there's no follow-up reply, so drop the placeholder.
      remove(reply.id);
      if (error.code === "action_expired") setActionStatus(actionId, "expired");
      // Handled elsewhere (another tab): reload to show what happened.
      if (error.code === "invalid_status") {
        void queryClient.invalidateQueries({ queryKey: messagesKey });
      }
      toast.error(decision === "confirm" ? "Couldn't confirm" : "Couldn't cancel", {
        description: error.message,
      });
    });
    setBusyAction(null);
  };

  return {
    isStreaming,
    busyAction,
    send,
    retry,
    stop: () => abortRef.current?.abort(),
    confirmAction: (actionId: string) => decide(actionId, "confirm"),
    cancelAction: (actionId: string) => decide(actionId, "cancel"),
  };
}
