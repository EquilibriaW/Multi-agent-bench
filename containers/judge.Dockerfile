FROM python:3.12.9-slim-bookworm

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    jq \
  && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash judge
RUN mkdir -p /workspace && chown -R judge:judge /workspace

USER judge
WORKDIR /workspace

ENTRYPOINT ["bash", "-lc", "sleep infinity"]
