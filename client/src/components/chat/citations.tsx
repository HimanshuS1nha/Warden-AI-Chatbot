import { FileTextIcon } from "lucide-react";

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";

import type { Citation } from "@/types/chat";

function location(citation: Citation): string {
  return [
    `v${citation.version}`,
    citation.section,
    citation.page !== null ? `p. ${citation.page}` : null,
  ]
    .filter(Boolean)
    .join(" · ");
}

type CitationListProps = {
  citations: Citation[];
  onCitationClick: (citation: Citation) => void;
};

// The sources under a reply, one chip per cited excerpt.
export function CitationList({ citations, onCitationClick }: CitationListProps) {
  if (citations.length === 0) return null;

  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-xs font-medium text-muted-foreground">Sources</span>
      <ul className="flex flex-wrap gap-1.5">
        {citations.map((citation) => (
          <li key={citation.n}>
            <button
              type="button"
              onClick={() => onCitationClick(citation)}
              className="flex max-w-72 items-center gap-1.5 rounded-md border bg-card px-2 py-1 text-left text-xs transition-colors hover:bg-muted"
            >
              <span className="flex size-4 shrink-0 items-center justify-center rounded-sm bg-secondary text-[0.65rem] font-medium text-secondary-foreground">
                {citation.n}
              </span>
              <span className="truncate font-medium">{citation.title}</span>
              <span className="shrink-0 text-muted-foreground">{citation.section}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

type CitationSheetProps = {
  citation: Citation | null;
  onOpenChange: (open: boolean) => void;
};

// The full excerpt behind a citation.
export function CitationSheet({ citation, onOpenChange }: CitationSheetProps) {
  return (
    <Sheet open={citation !== null} onOpenChange={onOpenChange}>
      <SheetContent className="w-full sm:max-w-md">
        {citation && (
          <>
            <SheetHeader>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <FileTextIcon className="size-3.5" />
                Source {citation.n}
              </div>
              <SheetTitle>{citation.title}</SheetTitle>
              <SheetDescription>{location(citation)}</SheetDescription>
            </SheetHeader>
            <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-4">
              <blockquote className="border-l-2 border-primary pl-3 text-sm leading-relaxed whitespace-pre-wrap">
                {citation.text}
              </blockquote>
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}
