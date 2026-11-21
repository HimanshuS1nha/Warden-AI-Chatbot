import { useLayoutEffect, useRef } from "react";

// How close to the bottom still counts as "at the bottom".
const BOTTOM_THRESHOLD = 80;

/**
 * Scroll behaviour for a chat log that grows at both ends:
 * - Starts at the bottom, and stays there as replies stream in, until the
 *   user scrolls up. Scrolling back down re-attaches it.
 * - When older messages load above, the messages on screen stay put.
 * - `pinToBottom()` jumps to the bottom and re-attaches (after sending).
 *
 * `messages` must be oldest first. The check runs after every render, since
 * any re-render of the log (a streamed token, a new message) can change its
 * height.
 */
export function useChatScroll(messages: readonly { id: string }[]) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const atBottom = useRef(true);
  const lastScrollHeight = useRef(0);
  const oldestId = messages[0]?.id;
  const lastOldestId = useRef(oldestId);

  // No dependency list on purpose: see above.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const olderLoaded = lastOldestId.current !== undefined && oldestId !== lastOldestId.current;
    if (atBottom.current) {
      el.scrollTop = el.scrollHeight;
    } else if (olderLoaded) {
      // Content grew above the viewport by this much; move down with it.
      el.scrollTop += el.scrollHeight - lastScrollHeight.current;
    }
    lastOldestId.current = oldestId;
    lastScrollHeight.current = el.scrollHeight;
  });

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    atBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < BOTTOM_THRESHOLD;
  };

  const pinToBottom = () => {
    atBottom.current = true;
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  };

  return { scrollRef, onScroll, pinToBottom };
}
