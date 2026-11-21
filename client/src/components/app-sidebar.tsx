import { MessageSquareIcon, PlusIcon, ShieldCheckIcon } from "lucide-react";

import { NavUser } from "@/components/nav-user";
import { Button } from "@/components/ui/button";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";

// Placeholder until the thread list is wired to the API (7.10).
const DUMMY_CONVERSATIONS = [
  { id: "1", title: "How many earned leaves do I have?" },
  { id: "2", title: "Work from home policy" },
  { id: "3", title: "Apply for sick leave tomorrow" },
  { id: "4", title: "Travel reimbursement limits" },
  { id: "5", title: "Who is my manager?" },
];

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
        <SidebarGroup>
          <SidebarGroupLabel>Conversations</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {DUMMY_CONVERSATIONS.map((conversation) => (
                <SidebarMenuItem key={conversation.id}>
                  {/* TODO(7.10): open the thread. */}
                  <SidebarMenuButton>
                    <MessageSquareIcon />
                    <span>{conversation.title}</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <NavUser />
      </SidebarFooter>
    </Sidebar>
  );
}
