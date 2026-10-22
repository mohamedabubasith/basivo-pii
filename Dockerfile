# basivo-pii REST service. The model is gated: request access on Hugging Face, pass your HF_TOKEN, and mount a volume
# on /hf to keep the download.
#   docker build -t basivo-pii .
#   docker run -p 8000:8000 -e BASIVO_PII_API_KEY=... -e BASIVO_PII_SECRET=... -e HF_TOKEN=... \
#     -v basivo-pii-hf:/hf --gpus all basivo-pii
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.22 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra server --no-install-project
COPY basivo_pii ./basivo_pii
COPY config.yaml ./
RUN uv sync --frozen --no-dev --extra server && useradd -m app && mkdir /hf && chown app /hf
USER app
ENV HF_HOME=/hf PATH=/app/.venv/bin:$PATH BASIVO_PII_CONFIG=/app/config.yaml
EXPOSE 8000
CMD ["uvicorn", "basivo_pii.server:app", "--host", "0.0.0.0", "--port", "8000"]
