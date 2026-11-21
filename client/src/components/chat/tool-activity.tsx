import { CircleCheckIcon, CircleXIcon, Loader2Icon } from "lucide-react";

import type { ToolCall } from "@/types/chat";

// Present-tense labels for the agent's tools (server/src/warden/graph/tools).
const TOOL_LABELS: Record<string, string> = {
  search_policies: "Searching company policies",
  get_my_profile: "Looking up your profile",
  get_my_manager: "Looking up your manager",
  get_my_leave_balance: "Checking your leave balance",
  get_my_leave_history: "Checking your leave history",
  get_my_attendance: "Checking your attendance",
  get_my_payslips: "Fetching your payslips",
  get_company_holidays: "Checking company holidays",
  list_my_team: "Looking up your team",
  get_team_leave_requests: "Checking your team's leave requests",
  get_team_attendance: "Checking your team's attendance",
  search_employees: "Searching employees",
  get_employee_profile: "Looking up the employee",
  get_employee_leave_summary: "Checking the employee's leave",
  apply_for_leave: "Preparing the leave application",
  cancel_my_leave_request: "Preparing the cancellation",
  approve_team_leave: "Preparing the approval",
  reject_team_leave: "Preparing the rejection",
};

function toolLabel(name: string): string {
  return TOOL_LABELS[name] ?? name.replaceAll("_", " ");
}

export function ToolActivity({ tools }: { tools: ToolCall[] }) {
  if (tools.length === 0) return null;

  return (
    <ul className="flex flex-col gap-1 text-xs text-muted-foreground">
      {tools.map((tool) => (
        <li key={tool.id} className="flex items-center gap-1.5">
          {tool.status === "running" ? (
            <Loader2Icon className="size-3.5 animate-spin" />
          ) : tool.status === "ok" ? (
            <CircleCheckIcon className="size-3.5 text-primary" />
          ) : (
            <CircleXIcon className="size-3.5 text-destructive" />
          )}
          <span>
            {toolLabel(tool.name)}
            {tool.status === "running" && "…"}
            {tool.status === "failed" && " failed"}
          </span>
        </li>
      ))}
    </ul>
  );
}
