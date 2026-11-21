import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { getErrorMessage } from "@/lib/api-error";
import { type MessagesData, updateAssistantMessage } from "@/lib/chat-cache";

import { rateMessage } from "@/api/chat";

import { QUERY_KEYS } from "@/constants/query-keys";

import type { Rating } from "@/types/chat";

type Variables = { messageId: string; rating: Rating | null };

/**
 * Thumbs up and down on the thread's replies. The button changes at once and
 * goes back if the request fails.
 */
export function useMessageFeedback(threadId: string) {
  const queryClient = useQueryClient();

  const setRating = (messageId: string, rating: Rating | null) =>
    updateAssistantMessage(queryClient, threadId, messageId, (m) => ({ ...m, feedback: rating }));

  const mutation = useMutation({
    mutationFn: ({ messageId, rating }: Variables) => rateMessage(messageId, rating),
    onMutate: ({ messageId, rating }: Variables) => {
      const previous = findRating(messageId);
      setRating(messageId, rating);
      return { previous };
    },
    onError: (error, { messageId }, context) => {
      if (context) setRating(messageId, context.previous);
      toast.error("Couldn't save your feedback", { description: getErrorMessage(error) });
    },
  });

  function findRating(messageId: string): Rating | null {
    const data = queryClient.getQueryData<MessagesData>(QUERY_KEYS.messages(threadId));
    const message = data?.pages.flatMap((p) => p.messages).find((m) => m.id === messageId);
    return message?.role === "assistant" ? message.feedback : null;
  }

  return (messageId: string, rating: Rating | null) => mutation.mutate({ messageId, rating });
}
