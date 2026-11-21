import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useMatchRoute, useNavigate } from "@tanstack/react-router";
import { toast } from "sonner";

import { getErrorMessage, isNotFound } from "@/lib/api-error";
import { updateThreads } from "@/lib/chat-cache";

import { deleteThread, renameThread } from "@/api/chat";

import { QUERY_KEYS } from "@/constants/query-keys";

import type { Thread } from "@/types/chat";

type Rename = { threadId: string; title: string; previousTitle: string | null };

const setTitle = (threadId: string, title: string | null) => (threads: Thread[]) =>
  threads.map((t) => (t.id === threadId ? { ...t, title } : t));

// Shows the new title at once and puts the old one back if the save fails.
export function useRenameThread() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ threadId, title }: Rename) => renameThread(threadId, title),
    onMutate: async ({ threadId, title }) => {
      // A list fetch in flight would bring the old title back.
      await queryClient.cancelQueries({ queryKey: QUERY_KEYS.threads });
      updateThreads(queryClient, setTitle(threadId, title));
    },
    // The server tidies whitespace, so show its version.
    onSuccess: (thread) => updateThreads(queryClient, setTitle(thread.id, thread.title)),
    onError: (error, { threadId, previousTitle }) => {
      updateThreads(queryClient, setTitle(threadId, previousTitle));
      toast.error("Couldn't rename the conversation", { description: getErrorMessage(error) });
    },
  });
}

// Deletes a thread and drops it from the caches. Leaves the page first if
// the thread is open, so its chat page doesn't try to reload it.
export function useDeleteThread() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const matchRoute = useMatchRoute();

  return useMutation({
    mutationFn: async (threadId: string) => {
      try {
        await deleteThread(threadId);
      } catch (error) {
        // Already deleted (in another tab): the outcome the user wanted.
        if (!isNotFound(error)) throw error;
      }
    },
    onSuccess: async (_, threadId) => {
      if (matchRoute({ to: "/chat/$threadId", params: { threadId } })) {
        await navigate({ to: "/", replace: true });
      }
      updateThreads(queryClient, (threads) => threads.filter((t) => t.id !== threadId));
      queryClient.removeQueries({ queryKey: QUERY_KEYS.messages(threadId) });
      toast.success("Conversation deleted");
    },
    onError: (error) => {
      toast.error("Couldn't delete the conversation", { description: getErrorMessage(error) });
    },
  });
}
