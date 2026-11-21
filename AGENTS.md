# Warden

An HR assistant chatbot. Employees ask about company policy (answered from PDFs with citations), read their own HR data, and apply for or approve leave through chat. The language model is untrusted: permissions are enforced by the backend, never by the model.

- `server/`: Python 3.12, FastAPI, SQLAlchemy (async) + PostgreSQL, LangGraph agent, Qdrant retrieval, Redis rate limits, Gemini. See `server/AGENTS.md`.
- `client/`: React 19, TypeScript, Vite, TanStack Router and Query, Tailwind CSS 4, shadcn on Base UI. See `client/AGENTS.md`.
- `docker-compose.yml` runs the whole app; `docker-compose.dev.yml` only the backing services for local development.
- `SECURITY.md` is the threat model. Keep it accurate when changing anything it describes.

Commit messages: `type(scope): what changed ... added`, e.g. `feat(server): redis rate limits for login and chat added`. Scope is `server` or `client`, or left out when a change spans both.
