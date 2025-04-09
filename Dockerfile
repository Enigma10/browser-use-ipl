FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libssl-dev \
    libffi-dev \
    openssl \
    ca-certificates \
    libnss3 \
    libnspr4 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libxkbcommon0 \
    libasound2 \
    libatspi2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files first
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install uv && \
    uv pip install --system -e .

# Install Playwright dependencies and browsers
RUN playwright install-deps && playwright install

# Expose the port that uvicorn will run on
EXPOSE 8111

# Command to run using uv
CMD ["uv", "run", "main.py"]
