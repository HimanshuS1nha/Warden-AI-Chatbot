import { useInfiniteQuery } from "@tanstack/react-query";
import { useParams } from "@tanstack/react-router";
import { RotateCwIcon } from "lucide-react";
import { useEffect } from "react";
import { useInView } from "react-intersection-observer";

import { ThreadItem } from "@/components/thread-item";
import { Button } from "@/components/ui/button";
import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuSkeleton,
} from "@/components/ui/sidebar";

import { getErrorMessage } from "@/lib/api-error";

import { threadsInfiniteQueryOptions } from "@/api/chat";

export function ThreadList() {
  const {
    data,
    error,
    isPending,
    isError,
    refetch,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
  } = useInfiniteQuery(threadsInfiniteQueryOptions);
  // Undefined outside a chat page.
  const { threadId: activeThreadId } = useParams({ strict: false });

  // Load the next page when the end of the list scrolls into view.
  const { ref: sentinelRef, inView } = useInView({ rootMargin: "200px" });
  useEffect(() => {
    if (inView && hasNextPage && !isFetchingNextPage) void fetchNextPage();
  }, [inView, hasNextPage, isFetchingNextPage, fetchNextPage]);

  const threads = data?.pages.flatMap((page) => page.threads) ?? [];

  return (
    <SidebarGroup>
      <SidebarGroupLabel>Conversations</SidebarGroupLabel>
      <SidebarGroupContent>
        {isPending ? (
          <ThreadSkeletons count={6} />
        ) : isError ? (
          <div className="flex flex-col items-start gap-2 px-2 py-1 text-sm text-muted-foreground">
            <p>Couldn't load conversations. {getErrorMessage(error)}</p>
            <Button variant="outline" size="sm" onClick={() => void refetch()}>
              <RotateCwIcon />
              Retry
            </Button>
          </div>
        ) : threads.length === 0 ? (
          <p className="px-2 py-1 text-sm text-muted-foreground">No conversations yet.</p>
        ) : (
          <SidebarMenu>
            {threads.map((thread) => (
              <ThreadItem key={thread.id} thread={thread} isActive={thread.id === activeThreadId} />
            ))}
            {isFetchingNextPage && <ThreadSkeletons count={2} />}
            {hasNextPage && <li ref={sentinelRef} aria-hidden className="h-px" />}
          </SidebarMenu>
        )}
      </SidebarGroupContent>
    </SidebarGroup>
  );
}

function ThreadSkeletons({ count }: { count: number }) {
  return Array.from({ length: count }, (_, i) => (
    <SidebarMenuItem key={i}>
      <SidebarMenuSkeleton showIcon />
    </SidebarMenuItem>
  ));
}
