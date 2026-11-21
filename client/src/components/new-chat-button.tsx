import { type InfiniteData, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { Loader2Icon, PlusIcon } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { useSidebar } from "@/components/ui/sidebar";

import { getErrorMessage } from "@/lib/api-error";

import { createThread } from "@/api/chat";

import { QUERY_KEYS } from "@/constants/query-keys";

import type { ThreadPage } from "@/types/chat";

export function NewChatButton() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const { isMobile, setOpenMobile } = useSidebar();

  const createThreadMutation = useMutation({
    mutationFn: createThread,
    onSuccess: async (thread) => {
      const threads = queryClient.getQueryData<InfiniteData<ThreadPage>>(QUERY_KEYS.threads);
      const [firstPage, ...otherPages] = threads?.pages ?? [];
      if (threads && firstPage) {
        // The list is newest first, so the new thread goes at the top. Adding
        // it here saves refetching every loaded page.
        queryClient.setQueryData<InfiniteData<ThreadPage>>(QUERY_KEYS.threads, {
          ...threads,
          pages: [{ ...firstPage, threads: [thread, ...firstPage.threads] }, ...otherPages],
        });
      } else {
        // The list hasn't loaded (or failed), so there's nothing to add to.
        void queryClient.invalidateQueries({ queryKey: QUERY_KEYS.threads });
      }
      if (isMobile) setOpenMobile(false);
      await navigate({ to: "/chat/$threadId", params: { threadId: thread.id } });
    },
    onError: (error) => {
      toast.error("Couldn't start a new chat", { description: getErrorMessage(error) });
    },
  });

  return (
    <Button
      variant="outline"
      className="w-full justify-start"
      disabled={createThreadMutation.isPending}
      onClick={() => createThreadMutation.mutate()}
    >
      {createThreadMutation.isPending ? <Loader2Icon className="animate-spin" /> : <PlusIcon />}
      New chat
    </Button>
  );
}
