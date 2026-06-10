# Use python:3.10-slim as the base image for a lightweight container
FROM python:3.10-slim

# Set environment variables
# Prevents Python from writing pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1
# Set default host and port for the FastAPI server
ENV HOST=0.0.0.0
ENV PORT=5000

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install PyTorch CPU-only version first to save image space,
# then install remaining requirements.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose port 5000 (standard port for the FastAPI app)
EXPOSE 5000

# Start the FastAPI application
CMD ["python", "app/app.py"]
