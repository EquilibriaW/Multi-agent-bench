FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Core utilities + Docker dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    build-essential \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CE
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

# Install standalone docker-compose v2 (some ABC scripts use `docker-compose` not `docker compose`)
RUN ln -sf /usr/libexec/docker/cli-plugins/docker-compose /usr/local/bin/docker-compose

# Python 3 + pip + common test deps
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager used by ABC test scripts)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && cp /root/.local/bin/uv /usr/local/bin/uv \
    && cp /root/.local/bin/uvx /usr/local/bin/uvx

# Allow user to run Docker without sudo
RUN usermod -aG docker user 2>/dev/null || true

# Start Docker daemon on sandbox boot and ensure socket is accessible
RUN mkdir -p /etc/e2b && printf '#!/bin/bash\ndockerd &\nfor i in $(seq 1 30); do [ -S /var/run/docker.sock ] && break; sleep 1; done\nchmod 666 /var/run/docker.sock\n' > /etc/e2b/start.sh \
    && chmod +x /etc/e2b/start.sh
