FROM python:3.11-slim

WORKDIR /app

ARG EMBEDDING_MODEL=intfloat/multilingual-e5-base

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Store HuggingFace model cache under /app so it is accessible to the
# non-root mesh user at runtime (chown -R mesh:mesh /app covers it).
ENV HF_HOME=/app/.cache

# Pre-download embedding model into image (~560 MB, cached in Docker layer).
# Eliminates 5-10 min cold start and allows air-gapped deployment.
# Write sentinel so runtime can verify the baked model matches.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
model_name = '${EMBEDDING_MODEL}'; \
print(f'Downloading model: {model_name}'); \
SentenceTransformer(model_name); \
print('Model cached successfully')" && \
    echo "${EMBEDDING_MODEL}" > /app/.cache/.baked-model-name

RUN groupadd -r mesh && useradd -r -g mesh -d /app mesh

COPY mesh/ ./mesh/
COPY ui/ ./ui/
COPY examples/ ./examples/

RUN chown -R mesh:mesh /app

USER mesh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["gunicorn", "mesh.main:app", "-k", "uvicorn.workers.UvicornWorker", \
    "--bind", "0.0.0.0:8000", \
    "--workers", "1", \
    "--graceful-timeout", "30", \
    "--timeout", "120"]
