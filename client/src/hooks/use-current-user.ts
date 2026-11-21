import { getRouteApi } from "@tanstack/react-router";

const rootLayoutRoute = getRouteApi("/_root");

// The signed-in user, from the /_root layout's loader. Only works in routes
// under _root/, where the loader has already redirected anyone signed out.
//
// Read from loader data rather than useSuspenseQuery(meQueryOptions): logout
// removes the `me` query, and a suspense query re-rendering mid-logout would
// refetch /auth/me and fail.
export function useCurrentUser() {
  return rootLayoutRoute.useLoaderData();
}
