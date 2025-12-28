FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir hatchling \
  && pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "agentic_rag_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
