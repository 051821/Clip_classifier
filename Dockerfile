# ── Base image ────────────────────────────────────────────────────────────────
# Use slim Python to keep the image small.
# If you have a GPU on the server, swap this for:
#   FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download CLIP model weights into the image ────────────────────────────
# This avoids downloading at runtime and makes the container self-contained.
RUN python -c "\
from transformers import CLIPModel, CLIPProcessor; \
CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
print('CLIP model cached.')"

# ── Copy application code ─────────────────────────────────────────────────────
COPY . .

# ── Output directory for prediction.csv ──────────────────────────────────────
RUN mkdir -p /output
ENV OUTPUT_CSV_PATH=/output/prediction.csv

# ── Environment variables (values provided at runtime, NOT hardcoded) ─────────
# DB
ENV DB_HOST=""
ENV DB_PORT="5432"
ENV DB_NAME=""
ENV DB_USER=""
ENV DB_PASSWORD=""

# AWS
ENV AWS_ACCESS_KEY_ID=""
ENV AWS_SECRET_ACCESS_KEY=""
ENV AWS_REGION="ap-south-1"
ENV S3_BUCKET_NAME=""

# CLIP tuning
ENV CONFIDENCE_THRESHOLD="0.65"
ENV BATCH_SIZE="16"
ENV LIMIT=""

# ── Run ───────────────────────────────────────────────────────────────────────
CMD ["python", "main.py"]
