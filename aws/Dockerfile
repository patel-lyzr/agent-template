# Bedrock AgentCore Runtime — must be ARM64 (Graviton only).
# DO NOT EDIT.
FROM --platform=linux/arm64 python:3.11-slim

WORKDIR /app

# System deps for common Python wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      git \
    && rm -rf /var/lib/apt/lists/*

# Combine cloud runtime deps with user deps, then install.
COPY aws/requirements.txt ./aws-requirements.txt
COPY requirements.txt     ./user-requirements.txt
RUN cat aws-requirements.txt user-requirements.txt > requirements.txt \
    && pip install --no-cache-dir -r requirements.txt

# Copy runtime wrapper and user code.
COPY aws/entrypoint.py ./entrypoint.py
COPY src/              ./src/

# Make user code importable.
ENV PYTHONPATH=/app/src

# AgentCore Runtime expects the app on port 8080.
EXPOSE 8080

# entrypoint.py boots BedrockAgentCoreApp and serves /invocations + /ping.
CMD ["python", "entrypoint.py"]
