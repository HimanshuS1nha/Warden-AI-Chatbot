# Warden client

The chat web app. React 19, TypeScript, Vite, TanStack Router (file-based) and TanStack Query, Tailwind CSS 4, shadcn components on Base UI, managed with pnpm.

```bash
pnpm dev                       # http://localhost:5173, API from VITE_API_URL (.env)
pnpm typecheck && pnpm lint && pnpm format:check
pnpm build
```

## Layout

```
src/
  main.tsx, router.tsx   app entry, router with the QueryClient in its context
  routes/                file-based routes; routeTree.gen.ts is generated, don't edit
    __root.tsx           providers, toaster, devtools
    login.tsx            login form with demo account shortcuts
    _root/route.tsx      signed-in layout: loader fetches /auth/me or redirects to /login
    _root/index.tsx      welcome screen when no thread is open
    _root/chat/$threadId.tsx   one thread
  api/                   request functions and query options per area (auth, chat)
  lib/
    axios.ts             axios instance: cookies, one refresh-and-retry on 401
    stream.ts            SSE reader for chat replies (fetch, not axios)
    chat-cache.ts        helpers that update threads and messages in the query cache
    api-error.ts         error messages from the server's {"message"} body
  hooks/                 use-chat-stream (send, confirm, cancel), thread mutations, feedback
  components/
    chat/                message, markdown, citations, tool activity, confirmation card, composer
    ui/                  shadcn components (add with the shadcn CLI, edit freely)
  constants/             API routes, query keys, roles, demo accounts
  types/                 API types; validators/  zod schemas for forms
```

## Design rules

- **The server decides everything.** The client never checks permissions or computes leave days; it shows what the API returns.
- **Auth is httpOnly cookies.** No tokens in JS or storage. Axios sends cookies and, on a 401, refreshes once (`refreshSession`, shared by parallel requests because the server rotates refresh tokens) and retries. `stream.ts` does the same for fetch.
- **Model output is untrusted.** `components/chat/markdown.tsx` drops raw HTML and images and renders only relative links. Don't loosen this: it stops tracking images and exfiltration links. Production nginx adds a CSP on top.
- **Chat streams into the query cache.** `useChatStream` writes token, tool, citation and confirmation events into the thread's messages cache (`lib/chat-cache.ts`), so the page renders from one source.
- **Confirmation cards are the only way to make a change.** Confirm and Cancel call `/chat/actions/{id}/confirm|cancel` and stream the agent's follow-up.
- **Use the shared constants.** API paths from `constants/api-routes.ts`, query keys from `constants/query-keys.ts`, never inline strings.

## Conventions

- Imports use the `@/` alias, grouped: libraries, `@/lib`, `@/constants`, then types.
- Server data goes through TanStack Query (`queryOptions` and `infiniteQueryOptions` in `api/`); no fetching in effects.
- Forms use TanStack Form with zod validators from `validators/`.
- Error text comes from `getErrorMessage`; 4xx errors aren't retried.
- Formatting is oxfmt and linting oxlint. Keep comments short and about why.
