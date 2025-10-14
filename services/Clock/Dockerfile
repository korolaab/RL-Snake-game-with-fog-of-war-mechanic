FROM python:3.11-slim

# Install system dependencies for shared memory and semaphores
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./

# Expose any necessary ports (none for Clock service)
# EXPOSE 

# Command to run the service
CMD ["python", "main.py"]