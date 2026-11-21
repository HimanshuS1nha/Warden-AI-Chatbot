import { CheckIcon, CopyIcon, RotateCwIcon, ShieldCheckIcon } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { CitationList, CitationSheet } from "@/components/chat/citations";
import { ConfirmationCard, type ConfirmationCardProps } from "@/components/chat/confirmation-card";
import { ChatMarkdown } from "@/components/chat/markdown";
import { ToolActivity } from "@/components/chat/tool-activity";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

import type { AssistantChatMessage, ChatMessage, Citation, UserChatMessage } from "@/types/chat";

type ActionHandlers = Omit<ConfirmationCardProps, "action">;

type AssistantMessageProps = ActionHandlers & {
  message: AssistantChatMessage;
  // Set when the reply failed and its question can be asked again.
  onRetry?: () => void;
  retryDisabled?: boolean;
};

type ChatMessageItemProps = Omit<AssistantMessageProps, "message"> & { message: ChatMessage };

export function ChatMessageItem({ message, ...props }: ChatMessageItemProps) {
  return message.role === "user" ? (
    <UserMessage message={message} />
  ) : (
    <AssistantMessage message={message} {...props} />
  );
}

function UserMessage({ message }: { message: UserChatMessage }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-secondary px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap text-secondary-foreground">
        {message.content}
      </div>
    </div>
  );
}

function AssistantMessage({
  message,
  onRetry,
  retryDisabled,
  ...actionHandlers
}: AssistantMessageProps) {
  const [openCitation, setOpenCitation] = useState<Citation | null>(null);
  const isStreaming = message.status === "streaming";
  const isThinking = isStreaming && message.content === "";

  return (
    <div className="flex gap-3">
      <span className="flex size-7 shrink-0 items-center justify-center rounded-md bg-primary text-primary-foreground">
        <ShieldCheckIcon className="size-4" />
      </span>

      <div className="flex min-w-0 flex-1 flex-col gap-3 pt-0.5">
        <ToolActivity tools={message.tools} />

        {isThinking && <ThinkingDots />}

        {message.content && (
          <div>
            <ChatMarkdown
              content={message.content}
              citations={message.citations}
              onCitationClick={setOpenCitation}
            />
            {isStreaming && (
              <span className="ml-0.5 inline-block h-4 w-1.5 translate-y-0.5 animate-pulse bg-foreground/70" />
            )}
          </div>
        )}

        {message.status === "error" && (
          <div className="flex flex-wrap items-center gap-2 rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm text-destructive">
            <span className="flex-1">{message.error}</span>
            {onRetry && (
              <Button size="sm" variant="outline" onClick={onRetry} disabled={retryDisabled}>
                <RotateCwIcon />
                Retry
              </Button>
            )}
          </div>
        )}

        {message.status === "stopped" && (
          <p className="text-xs text-muted-foreground italic">You stopped this reply.</p>
        )}

        <CitationList citations={message.citations} onCitationClick={setOpenCitation} />

        {message.action && <ConfirmationCard action={message.action} {...actionHandlers} />}

        {message.status === "done" && message.content && (
          <MessageActions content={message.content} />
        )}
      </div>

      <CitationSheet
        citation={openCitation}
        onOpenChange={(open) => !open && setOpenCitation(null)}
      />
    </div>
  );
}

function ThinkingDots() {
  return (
    <div className="flex h-5 items-center gap-1" aria-label="Thinking">
      {[0, 150, 300].map((delay) => (
        <span
          key={delay}
          className="size-1.5 animate-bounce rounded-full bg-muted-foreground/60"
          style={{ animationDelay: `${delay}ms` }}
        />
      ))}
    </div>
  );
}

function MessageActions({ content }: { content: string }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast.error("Couldn't copy the reply");
    }
  };

  // TODO(7.16): thumbs up and down, once the server has a feedback endpoint.
  return (
    <div className="-ml-1.5 flex items-center gap-0.5 text-muted-foreground">
      <Tooltip>
        <TooltipTrigger
          render={
            <Button
              variant="ghost"
              size="icon-xs"
              aria-label={copied ? "Copied" : "Copy"}
              onClick={() => void copy()}
            />
          }
        >
          {copied ? <CheckIcon /> : <CopyIcon />}
        </TooltipTrigger>
        <TooltipContent>{copied ? "Copied" : "Copy"}</TooltipContent>
      </Tooltip>
    </div>
  );
}
