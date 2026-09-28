# Evaluation harness for the visa work-rights RAG system.
#
# The image carries the corpus snapshot, the pre-built vector stores, the recorded
# runs and the test collection, so a scoring run needs no network and no GPU: it
# reads stored embeddings and stored answers. Only generation and re-embedding
# reach out to Ollama, which runs as a separate service with the GPU attached.
FROM python:3.12-slim

# Pinned so an image rebuilt in six months resolves the same interpreter patch level.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OLLAMA_HOST=http://ollama:11434

WORKDIR /app

# Dependencies first so a source edit does not invalidate the wheel layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY tests/ ./tests/
COPY data/ ./data/
COPY results/ ./results/
COPY README.md ./
COPY docker/entrypoint.sh /usr/local/bin/entrypoint

# The published results are copied aside at build time. `reproduce` re-derives the
# scoring outputs from the stored runs and diffs them against this copy, which is
# what makes the claim "these numbers reproduce" checkable rather than asserted.
RUN cp -r results /baseline-results && chmod +x /usr/local/bin/entrypoint

# curl is needed only to wait on the Ollama service before a generation run.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Runs unprivileged. /out is the only path the container is expected to write
# outside itself, and it is where a run deposits anything worth keeping.
RUN useradd --create-home --uid 10001 harness \
    && mkdir -p /out \
    && chown -R harness:harness /app /out /baseline-results
USER harness

ENTRYPOINT ["entrypoint"]
CMD ["tests"]
