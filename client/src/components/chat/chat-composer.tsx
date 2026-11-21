import { ArrowUpIcon, SquareIcon } from "lucide-react";
import { useState } from "react";

import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupText,
  InputGroupTextarea,
} from "@/components/ui/input-group";

// Matches MAX_MESSAGE_LENGTH in server/src/warden/services/chat.py.
const MAX_MESSAGE_LENGTH = 2000;
// Show the counter only when it starts to matter.
const COUNTER_FROM = MAX_MESSAGE_LENGTH - 200;

export function ChatComposer({ isStreaming }: { isStreaming: boolean }) {
  const [value, setValue] = useState("");
  const length = value.trim().length;
  const tooLong = length > MAX_MESSAGE_LENGTH;

  return (
    <form
      onSubmit={(e) => e.preventDefault()}
      className="mx-auto flex w-full max-w-3xl flex-col gap-2 px-4 pb-4"
    >
      <InputGroup className="rounded-xl bg-card shadow-xs">
        <InputGroupTextarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Ask about policies, your leave or your team…"
          aria-label="Message"
          aria-invalid={tooLong}
          rows={1}
          className="max-h-48 min-h-12 px-3.5 pt-3"
        />
        <InputGroupAddon align="block-end">
          <InputGroupText className="hidden text-xs font-normal sm:inline">
            Enter to send, Shift + Enter for a new line
          </InputGroupText>
          <div className="ml-auto flex items-center gap-2">
            {length >= COUNTER_FROM && (
              <InputGroupText
                className={tooLong ? "text-xs text-destructive" : "text-xs font-normal"}
              >
                {length}/{MAX_MESSAGE_LENGTH}
              </InputGroupText>
            )}
            {isStreaming ? (
              <InputGroupButton
                variant="outline"
                size="icon-sm"
                className="rounded-full"
                aria-label="Stop"
              >
                <SquareIcon className="fill-current" />
              </InputGroupButton>
            ) : (
              <InputGroupButton
                type="submit"
                variant="default"
                size="icon-sm"
                className="rounded-full"
                aria-label="Send"
                disabled={length === 0 || tooLong}
              >
                <ArrowUpIcon />
              </InputGroupButton>
            )}
          </div>
        </InputGroupAddon>
      </InputGroup>
      <p className="text-center text-xs text-muted-foreground">
        Warden can make mistakes. Check important details against the cited policy.
      </p>
    </form>
  );
}
