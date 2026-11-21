import { createFileRoute } from "@tanstack/react-router";

import { ChatComposer } from "@/components/chat/chat-composer";
import { ChatMessageItem } from "@/components/chat/chat-message";
import { DUMMY_MESSAGES } from "@/components/chat/dummy-messages";
import { ScrollArea } from "@/components/ui/scroll-area";

export const Route = createFileRoute("/_root/chat/$threadId")({
  component: ChatPage,
});

function ChatPage() {
  // TODO(7.11, 7.12): load the thread's messages and stream replies.
  const messages = DUMMY_MESSAGES;
  const last = messages.at(-1);
  const isStreaming = last?.role === "assistant" && last.status === "streaming";

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <ScrollArea className="min-h-0 flex-1">
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-4 py-6">
          {messages.map((message) => (
            <ChatMessageItem key={message.id} message={message} />
          ))}
        </div>
      </ScrollArea>
      <ChatComposer isStreaming={isStreaming} />
    </div>
  );
}
