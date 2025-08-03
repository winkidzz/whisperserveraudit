# Lean Dockerfile for WhisperCapRover
# Downloads Whisper model at runtime to reduce image size

FROM python:3.11-slim

# Install system dependencies for PyAudio, Whisper, and other audio libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    portaudio19-dev \
    python3-dev \
    build-essential \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    libsndfile1 \
    pkg-config \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create a non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 app

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages with all heavy dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/models /app/cache && \
    chown -R app:app /app

# Set environment variables for the base image
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV WHISPER_MODEL=base
ENV HOST=0.0.0.0
ENV PORT=80
ENV MAX_CONNECTIONS=10
ENV LOG_LEVEL=info
ENV WHISPER_CACHE_DIR=/app/cache

# Copy application code
COPY --chown=app:app server.py .

# Expose port
EXPOSE 80

# Health check for CapRover
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

# Switch to non-root user
USER app

# Run the server (Whisper model will be downloaded on first run)
CMD ["python", "server.py"] 