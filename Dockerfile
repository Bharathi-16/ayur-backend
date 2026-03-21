# ============================================
# AyurParam AI — Production Dockerfile
# ============================================
FROM python:3.14-slim

# Prevent interactive prompts during package install
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install critical system packages (some Python ML libraries need build tools)
RUN apt-get update && apt-get install -y \
    curl git build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (for Docker cache efficiency)
COPY requirements.txt .

# OPTIMIZATION: Railway/Render typically does not offer GPUs on basic plans.
# The default Linux PyTorch size is several GB because of bundled CUDA libraries.
# We explicitly install the CPU version of PyTorch first. This reduces size significantly!
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY app/ ./app/
COPY templates/ ./templates/
COPY main.py .
COPY gunicorn.conf.py .

# Create data directory for SQLite
RUN mkdir -p /app/data /app/logs

# Railway dynamically assigns the PORT environment variable.

CMD ["gunicorn", "--config", "gunicorn.conf.py", "main:app"]

