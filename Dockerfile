FROM python:3.11-slim

WORKDIR /app

# Install build deps for psycopg binary + sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Bake the embedding model in so it's not fetched over the network at runtime
COPY config.py .
RUN python -c "from sentence_transformers import SentenceTransformer; import config; SentenceTransformer(config.EMBEDDING_MODEL)"
ENV HF_HUB_OFFLINE=1

# Vendor the YouTube MCP server so npx doesn't re-resolve it at every cold start
RUN npm install -g @kirbah/mcp-youtube

COPY . .
RUN chmod +x deploy/cloudrun-start.sh

EXPOSE 8000

CMD uvicorn agent.server:app --host 0.0.0.0 --port ${PORT:-8000}
