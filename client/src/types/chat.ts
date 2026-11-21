// Mirror ThreadResponse and ThreadPage in server/src/warden/schemas/chat.py.
export type Thread = {
  id: string;
  // Null until the first message is sent.
  title: string | null;
  created_at: string;
};

export type ThreadPage = {
  threads: Thread[];
  // Pass as `before` to get the next page; null on the last one.
  next: string | null;
};
