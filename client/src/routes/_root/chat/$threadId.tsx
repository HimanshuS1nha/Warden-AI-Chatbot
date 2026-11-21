import { useInfiniteQuery } from "@tanstack/react-query";
import { Link, createFileRoute } from "@tanstack/react-router";
import { isAxiosError } from "axios";
import { Loader2Icon, MessageSquareOffIcon, RotateCwIcon } from "lucide-react";
import { useEffect } from "react";
import { useInView } from "react-intersection-observer";

import { ChatComposer } from "@/components/chat/chat-composer";
import { ChatEmptyState } from "@/components/chat/chat-empty-state";
import { ChatMessageItem } from "@/components/chat/chat-message";
import { Button } from "@/components/ui/button";
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { Skeleton } from "@/components/ui/skeleton";

import { useChatScroll } from "@/hooks/use-chat-scroll";
import { useChatStream } from "@/hooks/use-chat-stream";
import { useCurrentUser } from "@/hooks/use-current-user";
import { useMessageFeedback } from "@/hooks/use-message-feedback";
import { getErrorMessage } from "@/lib/api-error";

import { messagesInfiniteQueryOptions } from "@/api/chat";

export const Route = createFileRoute("/_root/chat/$threadId")({
  component: ChatPage,
});

function ChatPage() {
  const { threadId } = Route.useParams();
  // A fresh view per thread, so switching threads stops the old stream and
  // resets scroll state.
  return <ChatView key={threadId} threadId={threadId} />;
}

function ChatView({ threadId }: { threadId: string }) {
  const me = useCurrentUser();
  const {
    data,
    error,
    isPending,
    isError,
    refetch,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
  } = useInfiniteQuery(messagesInfiniteQueryOptions(threadId));
  const chat = useChatStream(threadId);
  const rate = useMessageFeedback(threadId);

  // Pages are newest first; the log reads oldest first, top to bottom.
  const messages = (data?.pages.flatMap((page) => page.messages) ?? []).toReversed();
  const { scrollRef, onScroll, pinToBottom } = useChatScroll(messages);

  // Load older messages when the top of the log scrolls into view.
  const { ref: olderRef, inView: olderInView } = useInView({ rootMargin: "300px 0px 0px 0px" });
  useEffect(() => {
    if (olderInView && hasNextPage && !isFetchingNextPage) void fetchNextPage();
  }, [olderInView, hasNextPage, isFetchingNextPage, fetchNextPage]);

  const send = (text: string) => {
    pinToBottom();
    void chat.send(text);
  };

  if (isError) {
    const notFound = isAxiosError(error) && error.response?.status === 404;
    return (
      <Empty className="flex-1">
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <MessageSquareOffIcon />
          </EmptyMedia>
          <EmptyTitle>
            {notFound ? "Conversation not found" : "Couldn't load this conversation"}
          </EmptyTitle>
          <EmptyDescription>
            {notFound ? "It may have been deleted, or the link is wrong." : getErrorMessage(error)}
          </EmptyDescription>
        </EmptyHeader>
        <EmptyContent>
          {notFound ? (
            <Button variant="outline" render={<Link to="/" />}>
              Back to home
            </Button>
          ) : (
            <Button variant="outline" onClick={() => void refetch()}>
              <RotateCwIcon />
              Try again
            </Button>
          )}
        </EmptyContent>
      </Empty>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div ref={scrollRef} onScroll={onScroll} className="relative min-h-0 flex-1 overflow-y-auto">
        {isFetchingNextPage && (
          // Overlaid so it doesn't push the messages down while it shows.
          <div className="pointer-events-none absolute inset-x-0 top-2 z-10 flex justify-center">
            <span className="flex items-center gap-1.5 rounded-full border bg-background px-3 py-1 text-xs text-muted-foreground shadow-xs">
              <Loader2Icon className="size-3 animate-spin" />
              Loading earlier messages
            </span>
          </div>
        )}

        {isPending ? (
          <MessagesSkeleton />
        ) : messages.length === 0 ? (
          <div className="flex h-full flex-col">
            <ChatEmptyState role={me.role} onPick={send} />
          </div>
        ) : (
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-4 py-6">
            {hasNextPage && <div ref={olderRef} aria-hidden className="h-px" />}
            {messages.map((message, index) => {
              const previous = messages[index - 1];
              const canRetry =
                message.role === "assistant" &&
                message.status === "error" &&
                previous?.role === "user";
              const busy =
                message.role === "assistant" &&
                message.action &&
                chat.busyAction?.id === message.action.id
                  ? chat.busyAction.decision
                  : null;
              return (
                <ChatMessageItem
                  key={message.id}
                  message={message}
                  onRetry={canRetry ? () => void chat.retry(message.id) : undefined}
                  retryDisabled={chat.isStreaming}
                  onConfirm={(actionId) => {
                    pinToBottom();
                    void chat.confirmAction(actionId);
                  }}
                  onCancel={(actionId) => {
                    pinToBottom();
                    void chat.cancelAction(actionId);
                  }}
                  onRate={rate}
                  busy={busy}
                  disabled={chat.isStreaming}
                />
              );
            })}
          </div>
        )}
      </div>

      <ChatComposer
        onSend={send}
        onStop={chat.stop}
        isStreaming={chat.isStreaming}
        disabled={isPending}
      />
    </div>
  );
}

function MessagesSkeleton() {
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-4 py-6" aria-hidden>
      <Skeleton className="ml-auto h-10 w-2/5 rounded-2xl" />
      <div className="flex gap-3">
        <Skeleton className="size-7 shrink-0 rounded-md" />
        <div className="flex flex-1 flex-col gap-2">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-4/5" />
          <Skeleton className="h-4 w-3/5" />
        </div>
      </div>
      <Skeleton className="ml-auto h-10 w-1/3 rounded-2xl" />
      <div className="flex gap-3">
        <Skeleton className="size-7 shrink-0 rounded-md" />
        <div className="flex flex-1 flex-col gap-2">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-2/3" />
        </div>
      </div>
    </div>
  );
}
