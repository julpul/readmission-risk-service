# Minimal image for the prediction API.
# The model artifact is NOT baked in - mount it at runtime (see README):
#   docker build -t readmission-risk-service .
#   docker run --rm -p 8000:8000 -v "$PWD/artifacts:/app/artifacts:ro" readmission-risk-service

FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.2 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first: this layer is cached until pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 app
USER app

ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_PATH=/app/artifacts/model.joblib

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "readmission.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
