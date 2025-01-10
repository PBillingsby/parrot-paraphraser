FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  wget \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

# Copy the inference script
COPY run_inference.py .

# Ensure the outputs directory exists and is writable
RUN mkdir -p /outputs && chmod 777 /outputs

# Set Python to unbuffered mode to see output immediately
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Disable HuggingFace telemetry
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV HF_MLFLOW_TRACKING_UI_HOST=none

# Default command
CMD ["python", "-u", "run_inference.py"]