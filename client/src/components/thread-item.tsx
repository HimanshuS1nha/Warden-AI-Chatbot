import { Link } from "@tanstack/react-router";
import {
  Loader2Icon,
  MessageSquareIcon,
  MoreHorizontalIcon,
  PencilIcon,
  Trash2Icon,
} from "lucide-react";
import { useRef, useState } from "react";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import {
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar";

import { useDeleteThread, useRenameThread } from "@/hooks/use-thread-mutations";

import { THREAD_TITLE_MAX_LENGTH } from "@/api/chat";

import type { Thread } from "@/types/chat";

const UNTITLED = "New chat";

// Spaces and line breaks collapse to single spaces, as the server stores it.
const tidyTitle = (title: string) => title.trim().replace(/\s+/g, " ");

// One conversation in the sidebar: a link to it, and a menu to rename or
// delete it (shown on hover with a mouse, always on touch screens).
export function ThreadItem({ thread, isActive }: { thread: Thread; isActive: boolean }) {
  const { isMobile, setOpenMobile } = useSidebar();
  const [isRenaming, setIsRenaming] = useState(false);
  const [isConfirmingDelete, setIsConfirmingDelete] = useState(false);
  // Rename was picked: the closing menu mustn't take focus from the title field.
  const renamePicked = useRef(false);
  const renameMutation = useRenameThread();
  const title = thread.title ?? UNTITLED;

  const finishRename = (newTitle: string | null) => {
    renamePicked.current = false;
    setIsRenaming(false);
    if (newTitle && newTitle !== thread.title) {
      renameMutation.mutate({ threadId: thread.id, title: newTitle, previousTitle: thread.title });
    }
  };

  return (
    <SidebarMenuItem>
      {isRenaming ? (
        <RenameField initialTitle={thread.title ?? ""} onDone={finishRename} />
      ) : (
        <>
          <SidebarMenuButton
            isActive={isActive}
            render={<Link to="/chat/$threadId" params={{ threadId: thread.id }} />}
            // On phones the sidebar is an overlay; close it to show the chat.
            onClick={() => isMobile && setOpenMobile(false)}
          >
            <MessageSquareIcon />
            <span className={thread.title ? undefined : "text-muted-foreground italic"}>
              {title}
            </span>
          </SidebarMenuButton>
          <DropdownMenu>
            <DropdownMenuTrigger render={<SidebarMenuAction showOnHover />}>
              <MoreHorizontalIcon />
              <span className="sr-only">Options for {title}</span>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" finalFocus={() => !renamePicked.current}>
              <DropdownMenuItem
                onClick={() => {
                  renamePicked.current = true;
                  setIsRenaming(true);
                }}
              >
                <PencilIcon />
                Rename
              </DropdownMenuItem>
              <DropdownMenuItem variant="destructive" onClick={() => setIsConfirmingDelete(true)}>
                <Trash2Icon />
                Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </>
      )}
      <DeleteThreadDialog
        thread={thread}
        open={isConfirmingDelete}
        onOpenChange={setIsConfirmingDelete}
      />
    </SidebarMenuItem>
  );
}

// Enter or leaving the field saves; Escape cancels. Calls onDone once, with
// the tidied title, or null to keep the old one.
function RenameField({
  initialTitle,
  onDone,
}: {
  initialTitle: string;
  onDone: (title: string | null) => void;
}) {
  const [value, setValue] = useState(initialTitle);
  const finished = useRef(false);

  const finish = (title: string | null) => {
    if (finished.current) return;
    finished.current = true;
    onDone(title && tidyTitle(title));
  };

  return (
    <Input
      ref={(input) => {
        input?.focus();
        input?.select();
      }}
      aria-label="Conversation title"
      value={value}
      maxLength={THREAD_TITLE_MAX_LENGTH}
      placeholder={UNTITLED}
      className="h-8 bg-background"
      onChange={(e) => setValue(e.target.value)}
      onBlur={() => finish(value)}
      onKeyDown={(e) => {
        if (e.key === "Enter" && !e.nativeEvent.isComposing) {
          e.preventDefault();
          finish(value);
        } else if (e.key === "Escape") {
          // Don't let it close the sidebar overlay on phones too.
          e.preventDefault();
          e.stopPropagation();
          finish(null);
        }
      }}
    />
  );
}

function DeleteThreadDialog({
  thread,
  open,
  onOpenChange,
}: {
  thread: Thread;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const deleteMutation = useDeleteThread();

  return (
    <AlertDialog
      open={open}
      // Stays open while deleting, so the outcome isn't missed.
      onOpenChange={(next) => !deleteMutation.isPending && onOpenChange(next)}
    >
      <AlertDialogContent size="sm">
        <AlertDialogHeader>
          <AlertDialogTitle>Delete this conversation?</AlertDialogTitle>
          <AlertDialogDescription>
            “{thread.title ?? UNTITLED}” and all its messages will be deleted. A change waiting for
            your confirmation in it can't be confirmed afterwards. This can't be undone.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={deleteMutation.isPending}>Cancel</AlertDialogCancel>
          <AlertDialogAction
            variant="destructive"
            disabled={deleteMutation.isPending}
            // On success the thread leaves the list, and this dialog with it.
            onClick={() => deleteMutation.mutate(thread.id)}
          >
            {deleteMutation.isPending && <Loader2Icon className="animate-spin" />}
            Delete
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
