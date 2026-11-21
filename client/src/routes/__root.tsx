import { createRootRouteWithContext, Outlet } from "@tanstack/react-router";
import { TanStackRouterDevtools } from "@tanstack/react-router-devtools";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";

import type { QueryClient } from "@tanstack/react-query";

const RootLayout = () => (
  <>
    <Outlet />
    <TanStackRouterDevtools position="bottom-left" />
    <ReactQueryDevtools buttonPosition="bottom-right" />
  </>
);

type RouteContextType = {
  queryClient: QueryClient;
};

export const Route = createRootRouteWithContext<RouteContextType>()({
  component: RootLayout,
});
