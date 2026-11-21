# syntax=docker/dockerfile:1

# Build the static app. API calls go to /api on the same origin, which nginx
# below proxies to the server, so cookies and CORS need nothing special.
FROM node:24-alpine AS build
RUN npm install --global pnpm@12.4.1
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN --mount=type=cache,target=/root/.local/share/pnpm/store \
    pnpm install --frozen-lockfile
COPY . .
ENV VITE_API_URL=/api
RUN pnpm build


# Serve it with nginx as a non-root user (listens on 8080).
FROM nginxinc/nginx-unprivileged:1.29-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
