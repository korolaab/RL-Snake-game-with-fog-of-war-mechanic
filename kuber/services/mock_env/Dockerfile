FROM python:3.11-slim

WORKDIR /app

# Install system dependencies first (rarely change)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer caching
COPY ./src/requirements.txt /app/requirements.txt

# Install Python dependencies (only rebuilds when requirements.txt changes)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code last (changes frequently)
COPY ./src /app

ENTRYPOINT []
