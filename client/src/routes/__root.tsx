import { createRootRouteWithContext, Outlet } from "@tanstack/react-router";
import { TanStackRouterDevtools } from "@tanstack/react-router-devtools";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";

import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";

import type { QueryClient } from "@tanstack/react-query";

const RootLayout = () => (
  <TooltipProvider>
    <Outlet />
    <Toaster />
    <TanStackRouterDevtools position="bottom-left" />
    <ReactQueryDevtools buttonPosition="bottom-right" />
  </TooltipProvider>
);

type RouteContextType = {
  queryClient: QueryClient;
};

export const Route = createRootRouteWithContext<RouteContextType>()({
  component: RootLayout,
});
