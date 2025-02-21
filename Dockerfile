# Stage 1: Build Stage
FROM python:3.12.8-slim as builder
WORKDIR /app

# Copy requirements and install them (gunicorn should be in requirements.txt)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Stage 2: Final Image
FROM python:3.12.8-slim
WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY . .

# (Optional) Set environment variables for your app, if needed
ENV PATH="/usr/local/bin:${PATH}"

# Set the command to run gunicorn, adjust "project.wsgi" to your actual WSGI module
CMD ["gunicorn", "--worker-tmp-dir", "/dev/shm", "project.wsgi"]
