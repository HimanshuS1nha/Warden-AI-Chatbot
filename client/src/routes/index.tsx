import { useSuspenseQuery } from "@tanstack/react-query";
import { createFileRoute, redirect } from "@tanstack/react-router";

import { meQueryOptions } from "@/api/auth";

export const Route = createFileRoute("/")({
  loader: async ({ context: { queryClient } }) => {
    try {
      await queryClient.query(meQueryOptions);
    } catch {
      throw redirect({ to: "/login", replace: true });
    }
  },
  component: Index,
});

function Index() {
  // The loader has already filled the cache, so this never suspends.
  const { data: me } = useSuspenseQuery(meQueryOptions);

  return (
    <div className="p-2">
      <h3>Welcome, {me.name}!</h3>
    </div>
  );
}
