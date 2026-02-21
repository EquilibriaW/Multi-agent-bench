FROM python:3.12.9-slim-bookworm

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    jq \
    docker.io \
    docker-compose-plugin \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /runner
ENTRYPOINT ["bash", "-lc", "sleep infinity"]
