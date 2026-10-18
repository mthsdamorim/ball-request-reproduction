# Clean-room Linux run of the whole reproduction.
#   docker build -t ball-request-lab .
#   docker run --rm ball-request-lab
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv
WORKDIR /repo
COPY . .
RUN uv sync --locked
CMD ["uv", "run", "ball-request-lab", "all"]
