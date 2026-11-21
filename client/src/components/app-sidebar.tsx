import { ShieldCheckIcon } from "lucide-react";

import { NavUser } from "@/components/nav-user";
import { NewChatButton } from "@/components/new-chat-button";
import { ThreadList } from "@/components/thread-list";
import { Sidebar, SidebarContent, SidebarFooter, SidebarHeader } from "@/components/ui/sidebar";

export function AppSidebar() {
  return (
    <Sidebar>
      <SidebarHeader className="gap-3">
        <div className="flex items-center gap-2 px-2 pt-1 font-semibold">
          <span className="flex size-7 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <ShieldCheckIcon className="size-4" />
          </span>
          Warden
        </div>
        <NewChatButton />
      </SidebarHeader>

      <SidebarContent>
        <ThreadList />
      </SidebarContent>

      <SidebarFooter>
        <NavUser />
      </SidebarFooter>
    </Sidebar>
  );
}
