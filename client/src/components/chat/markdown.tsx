import { createContext, use, useMemo } from "react";
import Markdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import type { Citation } from "@/types/chat";

const CITATION_HREF = "#citation-";
// Same pattern as the server's citation markers: [1] or [1, 2], not a link.
const MARKER = /\[(\d+(?:\s*,\s*\d+)*)\](?!\()/g;

// Turn each [n] marker into a link the `a` component below renders as a chip.
function linkCitations(text: string): string {
  return text.replace(MARKER, (_, numbers: string) =>
    numbers
      .split(",")
      .map((n) => `[${n.trim()}](${CITATION_HREF}${n.trim()})`)
      .join(""),
  );
}

type CitationContextValue = {
  citations: Citation[];
  onCitationClick: (citation: Citation) => void;
};

// Lets the module-level `a` component reach this message's citations.
const CitationContext = createContext<CitationContextValue>({
  citations: [],
  onCitationClick: () => {},
});

function MarkdownLink({ href, children }: { href?: string; children?: React.ReactNode }) {
  const { citations, onCitationClick } = use(CitationContext);

  if (href?.startsWith(CITATION_HREF)) {
    const n = Number(href.slice(CITATION_HREF.length));
    const citation = citations.find((c) => c.n === n);
    if (!citation) return null;
    return (
      <button
        type="button"
        onClick={() => onCitationClick(citation)}
        className="mx-0.5 inline-flex h-4.5 min-w-4.5 -translate-y-0.5 items-center justify-center rounded-sm bg-secondary px-1 align-middle text-[0.7rem] font-medium text-secondary-foreground transition-colors hover:bg-primary hover:text-primary-foreground"
        aria-label={`Source ${n}: ${citation.title}`}
      >
        {children}
      </button>
    );
  }
  if (href?.startsWith("/") && !href.startsWith("//")) {
    return (
      <a href={href} className="font-medium text-primary underline underline-offset-4">
        {children}
      </a>
    );
  }
  return <span>{children}</span>;
}

const COMPONENTS: Components = {
  img: () => null,
  a: MarkdownLink,
  p: ({ children }) => <p className="leading-relaxed not-first:mt-3">{children}</p>,
  ul: ({ children }) => <ul className="mt-3 ml-5 list-disc space-y-1">{children}</ul>,
  ol: ({ children }) => <ol className="mt-3 ml-5 list-decimal space-y-1">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
  h1: ({ children }) => <h3 className="mt-4 font-semibold first:mt-0">{children}</h3>,
  h2: ({ children }) => <h3 className="mt-4 font-semibold first:mt-0">{children}</h3>,
  h3: ({ children }) => <h3 className="mt-4 font-semibold first:mt-0">{children}</h3>,
  code: ({ children }) => (
    <code className="rounded bg-muted px-1 py-0.5 font-mono text-[0.85em]">{children}</code>
  ),
  pre: ({ children }) => (
    <pre className="mt-3 overflow-x-auto rounded-md bg-muted p-3 text-sm [&>code]:bg-transparent [&>code]:p-0">
      {children}
    </pre>
  ),
  blockquote: ({ children }) => (
    <blockquote className="mt-3 border-l-2 pl-3 text-muted-foreground">{children}</blockquote>
  ),
  table: ({ children }) => (
    <div className="mt-3 overflow-x-auto rounded-md border">
      <table className="w-full text-sm">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-muted/50">{children}</thead>,
  tr: ({ children }) => <tr className="border-b last:border-0">{children}</tr>,
  th: ({ children }) => <th className="px-3 py-2 text-left font-medium">{children}</th>,
  td: ({ children }) => <td className="px-3 py-2">{children}</td>,
};

type Props = CitationContextValue & { content: string };

// Model output is untrusted: raw HTML is dropped, images never load (no
// tracking pixels or exfiltration URLs), and only relative links render.
export function ChatMarkdown({ content, citations, onCitationClick }: Props) {
  const citationContext = useMemo(
    () => ({ citations, onCitationClick }),
    [citations, onCitationClick],
  );

  return (
    <CitationContext value={citationContext}>
      <div className="text-sm">
        <Markdown remarkPlugins={[remarkGfm]} skipHtml components={COMPONENTS}>
          {linkCitations(content)}
        </Markdown>
      </div>
    </CitationContext>
  );
}
