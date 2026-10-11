# Official Python 3.12 Slim base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install required Linux audio/vision runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Environment configuration
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# Expose Hugging Face Spaces container port
EXPOSE 7860

# Launch NeuroCare production server
CMD ["python", "app.py"]
