FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
# O wheel Linux do XGBoost puxa o NCCL (~200 MB) para treino distribuido em GPU;
# a inferencia aqui e so CPU.
RUN pip install -r requirements.txt \
    && pip uninstall -y nvidia-nccl-cu12


FROM python:3.12-slim AS runtime

# libgomp1: runtime OpenMP exigido pelo XGBoost
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --uid 10001 app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/predictive_model.joblib

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY src ./src
RUN mkdir -p models && chown app:app models

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
