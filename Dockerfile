FROM python:3.11-slim

WORKDIR /app

# Install system libraries needed for LightGBM (OpenMP) and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run training to generate serialized models and datasets
RUN python src/train.py

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV API_PORT=8000
ENV API_HOST=0.0.0.0

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
