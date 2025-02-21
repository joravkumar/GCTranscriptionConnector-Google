# -------------------------------------------------
# 1) Use a suitable Python base image
# -------------------------------------------------
FROM python:3.12.8-slim

# -------------------------------------------------
# 2) Install system packages needed for FFmpeg, etc.
#    (since you used to rely on Aptfile)
# -------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libvpx7 \
    pulseaudio \
    libmp3lame0 \
    libpulse0 \
    libpulse-dev \
  && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------
# 3) Create a working directory
# -------------------------------------------------
WORKDIR /app

# -------------------------------------------------
# 4) Copy and install Python dependencies
# -------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------
# 5) Copy your entire codebase into the container
# -------------------------------------------------
COPY . .

# -------------------------------------------------
# 6) Expose the port you actually listen on
#    (Your code listens on GENESYS_LISTEN_PORT=443, so we EXPOSE 443)
# -------------------------------------------------
EXPOSE 443

# -------------------------------------------------
# 7) Finally, run your main Python script
# -------------------------------------------------
CMD ["python", "oai_middleware.py"]
