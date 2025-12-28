FROM node:20-alpine

WORKDIR /app

COPY package.json pnpm-workspace.yaml pnpm-lock.yaml ./
COPY frontend/package.json ./frontend/package.json

RUN corepack enable \
  && corepack prepare pnpm@9.1.0 --activate \
  && pnpm install --filter agentic-rag-frontend... --frozen-lockfile

COPY frontend ./frontend

EXPOSE 3000

WORKDIR /app/frontend

CMD ["pnpm", "dev", "-H", "0.0.0.0", "-p", "3000"]
