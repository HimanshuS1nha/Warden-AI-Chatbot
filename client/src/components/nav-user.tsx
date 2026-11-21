import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useRouter } from "@tanstack/react-router";
import { ChevronsUpDownIcon, LogOutIcon } from "lucide-react";
import { toast } from "sonner";

import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { SidebarMenu, SidebarMenuButton, SidebarMenuItem } from "@/components/ui/sidebar";

import { useCurrentUser } from "@/hooks/use-current-user";
import { getErrorMessage } from "@/lib/api-error";

import { logout } from "@/api/auth";

import { QUERY_KEYS } from "@/constants/query-keys";
import { ROLE_LABELS } from "@/constants/roles";

function initials(name: string): string {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((word) => word[0]?.toUpperCase())
    .join("");
}

export function NavUser() {
  const me = useCurrentUser();
  const queryClient = useQueryClient();
  const router = useRouter();
  const navigate = useNavigate();

  const logoutMutation = useMutation({
    mutationFn: logout,
    onSuccess: async () => {
      // Drop `me` first, or the /login loader would find it and send the
      // user straight back here.
      queryClient.removeQueries({ queryKey: QUERY_KEYS.me });
      await navigate({ to: "/login", replace: true });
      // Clear everything else only once this layout has unmounted, so no
      // mounted query refetches without a session. Clearing the router cache
      // stops the next user from briefly seeing this user's loader data.
      queryClient.clear();
      router.clearCache();
    },
    onError: (error) => {
      toast.error("Couldn't log out", { description: getErrorMessage(error) });
    },
  });

  const avatar = (
    <Avatar className="rounded-lg">
      <AvatarFallback className="rounded-lg bg-primary text-xs text-primary-foreground">
        {initials(me.name)}
      </AvatarFallback>
    </Avatar>
  );

  const identity = (
    <div className="grid flex-1 text-left text-sm leading-tight">
      <span className="truncate font-medium">{me.name}</span>
      <span className="truncate text-xs text-muted-foreground">{me.email}</span>
    </div>
  );

  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <DropdownMenu>
          <DropdownMenuTrigger
            render={
              <SidebarMenuButton
                size="lg"
                className="data-popup-open:bg-sidebar-accent data-popup-open:text-sidebar-accent-foreground"
              />
            }
          >
            {avatar}
            {identity}
            <ChevronsUpDownIcon className="ml-auto" />
          </DropdownMenuTrigger>
          <DropdownMenuContent side="top" align="end" className="w-(--anchor-width) min-w-56">
            <DropdownMenuGroup>
              <DropdownMenuLabel className="flex items-center gap-2 font-normal text-foreground">
                {avatar}
                {identity}
              </DropdownMenuLabel>
              <p className="px-2 pb-1.5 text-xs text-muted-foreground">
                {ROLE_LABELS[me.role]} · {me.job_title}
              </p>
            </DropdownMenuGroup>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              variant="destructive"
              disabled={logoutMutation.isPending}
              onClick={() => logoutMutation.mutate()}
            >
              <LogOutIcon />
              Log out
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </SidebarMenuItem>
    </SidebarMenu>
  );
}
