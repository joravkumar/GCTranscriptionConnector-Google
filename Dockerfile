# Use an official Python 3.12 slim image as the base.
FROM python:3.12-slim

# Install ffmpeg and its dependencies.
# The list below installs ffmpeg and the required libraries.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libvpx7 \
    pulseaudio \
    libmp3lame0 \
    libpulse0 \
    libpulse-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container.
WORKDIR /app

# (Optional) If you want to ensure Python output is not buffered:
ENV PYTHONUNBUFFERED=1

# Copy requirements.txt first to leverage Docker caching.
COPY requirements.txt .

# Install Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code.
COPY . .

# (Optional) If your app uses Gunicorn, and as the DO article explains,
# you must specify a temporary directory for Gunicorn workers.
# Replace "project.wsgi" with your actual WSGI module.
CMD ["gunicorn", "--worker-tmp-dir", "/dev/shm", "project.wsgi"]
