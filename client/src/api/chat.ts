import { infiniteQueryOptions } from "@tanstack/react-query";

import { axios } from "@/lib/axios";

import { API_ROUTES } from "@/constants/api-routes";
import { QUERY_KEYS } from "@/constants/query-keys";

import type { Thread, ThreadPage } from "@/types/chat";

const THREADS_PAGE_SIZE = 30;

export async function createThread(): Promise<Thread> {
  const { data } = await axios.post<Thread>(API_ROUTES.chat.threads);
  return data;
}

export async function getThreads(before: string | null): Promise<ThreadPage> {
  const { data } = await axios.get<ThreadPage>(API_ROUTES.chat.threads, {
    params: { limit: THREADS_PAGE_SIZE, ...(before && { before }) },
  });
  return data;
}

// The caller's threads, newest first, one cursor page at a time.
export const threadsInfiniteQueryOptions = infiniteQueryOptions({
  queryKey: QUERY_KEYS.threads,
  queryFn: ({ pageParam }) => getThreads(pageParam),
  initialPageParam: null as string | null,
  getNextPageParam: (lastPage) => lastPage.next,
});
