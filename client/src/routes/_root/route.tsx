import { Outlet, createFileRoute, redirect } from "@tanstack/react-router";

import { AppSidebar } from "@/components/app-sidebar";
import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";

import { meQueryOptions } from "@/api/auth";

// Pathless layout for every signed-in page. The loader guarantees a user, so
// child routes can read it with useCurrentUser() without handling "no user".
export const Route = createFileRoute("/_root")({
  loader: async ({ context: { queryClient } }) => {
    try {
      return await queryClient.query(meQueryOptions);
    } catch {
      throw redirect({ to: "/login", replace: true });
    }
  },
  component: RootLayout,
});

function RootLayout() {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset className="h-svh overflow-hidden">
        <header className="flex h-12 shrink-0 items-center border-b px-2">
          <SidebarTrigger />
        </header>
        <Outlet />
      </SidebarInset>
    </SidebarProvider>
  );
}
