FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY app.py cli.py ./
COPY data/raw ./data/raw

# Ingest at container start (index/graph/db live in the container filesystem;
# mount a volume at /app/data/processed to persist across restarts), then serve.
EXPOSE 8501
CMD ["sh", "-c", "python cli.py ingest && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
