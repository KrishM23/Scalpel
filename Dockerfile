FROM python:3.12-slim

WORKDIR /app

# CPU-only torch keeps the image small; override for GPU builds.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 scalpel \
    && mkdir -p /data/artifacts \
    && chown -R scalpel:scalpel /data /app

ENV SCALPEL_ARTIFACT_DIR=/data/artifacts \
    SCALPEL_DB_PATH=/data/scalpel.db \
    SCALPEL_REQUIRE_API_KEYS=1 \
    PYTHONUNBUFFERED=1

VOLUME /data
EXPOSE 8000
USER scalpel

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4)"

CMD ["scalpel", "serve", "--host", "0.0.0.0", "--port", "8000"]
