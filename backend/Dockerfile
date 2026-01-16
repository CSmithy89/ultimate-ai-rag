FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Playwright and general utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Playwright dependencies
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    # Additional utilities
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir hatchling \
  && pip install --no-cache-dir .

# Install Playwright browsers (chromium only to reduce image size)
RUN playwright install chromium

EXPOSE 8000

CMD ["uvicorn", "agentic_rag_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
