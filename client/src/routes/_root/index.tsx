import { createFileRoute } from "@tanstack/react-router";

import { useCurrentUser } from "@/hooks/use-current-user";

export const Route = createFileRoute("/_root/")({
  component: Index,
});

function Index() {
  const me = useCurrentUser();

  return (
    <div className="p-2">
      <h3>Welcome, {me.name}!</h3>
    </div>
  );
}
