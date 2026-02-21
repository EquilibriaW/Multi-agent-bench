FROM python:3.12.9-slim-bookworm

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    jq \
    make \
    gcc \
    g++ \
  && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash agent
RUN mkdir -p /workspace && chown -R agent:agent /workspace

USER agent
WORKDIR /workspace

ENTRYPOINT ["bash", "-lc", "sleep infinity"]
