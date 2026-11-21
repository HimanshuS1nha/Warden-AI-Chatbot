import { createFileRoute } from "@tanstack/react-router";
import { MessagesSquareIcon, PanelLeftIcon } from "lucide-react";

import { NewChatButton } from "@/components/new-chat-button";
import { Button } from "@/components/ui/button";
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { useSidebar } from "@/components/ui/sidebar";

import { useCurrentUser } from "@/hooks/use-current-user";

export const Route = createFileRoute("/_root/")({
  component: Index,
});

function Index() {
  const me = useCurrentUser();
  const { isMobile, state, toggleSidebar } = useSidebar();
  const firstName = me.name.split(" ")[0];
  // The conversation list is only on screen when the sidebar is open on desktop.
  const sidebarHidden = isMobile || state === "collapsed";

  return (
    <div className="flex flex-1 items-center justify-center p-6">
      <Empty>
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <MessagesSquareIcon />
          </EmptyMedia>
          <EmptyTitle>Welcome, {firstName}</EmptyTitle>
          <EmptyDescription>
            Ask about company policies, your leave or your team. Pick up a conversation from the
            sidebar, or start a new one.
          </EmptyDescription>
        </EmptyHeader>
        <EmptyContent>
          <div className="flex flex-wrap justify-center gap-2">
            <NewChatButton />
            {sidebarHidden && (
              <Button variant="outline" onClick={toggleSidebar}>
                <PanelLeftIcon />
                Browse conversations
              </Button>
            )}
          </div>
        </EmptyContent>
      </Empty>
    </div>
  );
}
