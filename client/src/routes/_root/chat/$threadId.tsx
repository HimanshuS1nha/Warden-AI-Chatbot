import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/_root/chat/$threadId")({
  component: ChatPage,
});

function ChatPage() {
  const { threadId } = Route.useParams();

  return (
    <div className="p-2">
      <h3>Hello from chat {threadId}</h3>
    </div>
  );
}
