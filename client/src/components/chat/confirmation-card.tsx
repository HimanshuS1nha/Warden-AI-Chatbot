import {
  CalendarCheckIcon,
  CalendarPlusIcon,
  CalendarXIcon,
  CircleAlertIcon,
  CircleCheckIcon,
  CircleSlashIcon,
  ClockIcon,
  type LucideIcon,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

import type { PendingAction, PendingActionStatus, PendingActionType } from "@/types/chat";

const ACTION_TITLES: Record<PendingActionType, { title: string; icon: LucideIcon }> = {
  apply_leave: { title: "Apply for leave", icon: CalendarPlusIcon },
  cancel_leave: { title: "Cancel leave", icon: CalendarXIcon },
  approve_leave: { title: "Approve leave", icon: CalendarCheckIcon },
  reject_leave: { title: "Reject leave", icon: CalendarXIcon },
};

const STATUS_BADGES: Record<
  Exclude<PendingActionStatus, "pending">,
  { label: string; icon: LucideIcon; variant: "secondary" | "outline" | "destructive" }
> = {
  confirmed: { label: "Confirmed", icon: CircleCheckIcon, variant: "secondary" },
  cancelled: { label: "Cancelled", icon: CircleSlashIcon, variant: "outline" },
  expired: { label: "Expired", icon: ClockIcon, variant: "outline" },
  failed: { label: "Didn't go through", icon: CircleAlertIcon, variant: "destructive" },
};

// A change the agent proposed. Nothing happens until the user confirms.
export function ConfirmationCard({ action }: { action: PendingAction }) {
  const { title, icon: Icon } = ACTION_TITLES[action.type];
  const badge = action.status === "pending" ? null : STATUS_BADGES[action.status];

  return (
    <div
      data-status={action.status}
      className="flex flex-col gap-3 rounded-lg border bg-card p-4 shadow-xs data-[status=pending]:border-primary/40 data-[status=pending]:ring-3 data-[status=pending]:ring-primary/10"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2 text-sm font-medium">
          <span className="flex size-7 items-center justify-center rounded-md bg-secondary text-secondary-foreground">
            <Icon className="size-4" />
          </span>
          {title}
        </div>
        {badge && (
          <Badge variant={badge.variant}>
            <badge.icon />
            {badge.label}
          </Badge>
        )}
      </div>

      <p className="text-sm leading-relaxed">{action.summary}</p>

      {action.status === "pending" && (
        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm">Confirm</Button>
          <Button size="sm" variant="outline">
            Cancel
          </Button>
          <span className="text-xs text-muted-foreground">Nothing changes until you confirm.</span>
        </div>
      )}
    </div>
  );
}
