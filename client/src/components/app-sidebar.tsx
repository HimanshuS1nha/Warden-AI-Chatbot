import { PlusIcon, ShieldCheckIcon } from "lucide-react";

import { NavUser } from "@/components/nav-user";
import { ThreadList } from "@/components/thread-list";
import { Button } from "@/components/ui/button";
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
        {/* TODO(7.10): create a thread. */}
        <Button variant="outline" className="w-full justify-start">
          <PlusIcon />
          New chat
        </Button>
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
