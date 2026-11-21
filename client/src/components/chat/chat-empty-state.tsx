import { MessageSquarePlusIcon } from "lucide-react";

import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";

import type { Role } from "@/types/employee";

// Starter questions that match what each role's tools can do.
const SUGGESTIONS: Record<Role, string[]> = {
  employee: [
    "What's my leave balance?",
    "How many earned leaves can I carry forward?",
    "Apply for casual leave tomorrow",
    "What are the company holidays this year?",
  ],
  manager: [
    "Show my team's pending leave",
    "Who in my team is on leave this week?",
    "What's my leave balance?",
    "What does the manager handbook say about approving leave?",
  ],
  hr: [
    "Find employees in Engineering",
    "Show the leave summary for an employee",
    "What does the disciplinary procedure say about warnings?",
    "What's my leave balance?",
  ],
};

type Props = {
  role: Role;
  onPick: (question: string) => void;
};

export function ChatEmptyState({ role, onPick }: Props) {
  return (
    <Empty className="flex-1">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <MessageSquarePlusIcon />
        </EmptyMedia>
        <EmptyTitle>Start the conversation</EmptyTitle>
        <EmptyDescription>Ask anything below, or try one of these.</EmptyDescription>
      </EmptyHeader>
      <EmptyContent className="max-w-md">
        <div className="flex flex-wrap justify-center gap-2">
          {SUGGESTIONS[role].map((question) => (
            <button
              key={question}
              type="button"
              onClick={() => onPick(question)}
              className="rounded-full border bg-card px-3 py-1.5 text-sm transition-colors hover:bg-muted"
            >
              {question}
            </button>
          ))}
        </div>
      </EmptyContent>
    </Empty>
  );
}
