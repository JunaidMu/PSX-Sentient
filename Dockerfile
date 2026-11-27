# One image for every PSX-Sentient service (API, dashboard, training/retraining jobs, MLflow server).
#   docker build -t psx-sentient .                         # core
#   docker build --build-arg INSTALL_NLP=true -t psx-sentient .   # + FinBERT (torch, ~1 GB more)
FROM python:3.12-slim

ARG INSTALL_NLP=false
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PYTHONWARNINGS=ignore

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-nlp.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt && \
    if [ "$INSTALL_NLP" = "true" ]; then \
      pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-nlp.txt; fi

COPY . .

EXPOSE 8000 8501 5000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
