FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==2.14.1

WORKDIR /mlflow

EXPOSE 5000

# Backend store: SQLite (fine for solo dev, zero external DB dependency).
# Artifact store: local volume mounted from docker-compose.
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/data/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
