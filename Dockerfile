# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install system dependencies needed by some Python packages (e.g., pydub/ffmpeg if later required)
# Uncomment if you add audio processing that requires ffmpeg
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ffmpeg \
#  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default port used by the app
EXPOSE 8080

# Use a non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Run the server
CMD ["python", "main.py"]
