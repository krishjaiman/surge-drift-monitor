# Small, purpose-built image: reads parquet/jsonl from a read-only mount,
# exposes Prometheus gauges. Nothing else runs in this container.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY exporter.py .

EXPOSE 8000

CMD ["python", "exporter.py"]