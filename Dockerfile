FROM python:3.12-slim AS base

WORKDIR /app

# build tools needed by lightgbm wheels on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY app ./app
COPY scripts ./scripts
COPY data ./data

RUN pip install --no-cache-dir -e .[app]

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -f http://localhost:8501/api/health || exit 1

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8501"]
