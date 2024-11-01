# Base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /openvino

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire app directory into the Docker container
COPY app/requirements.txt /openvino/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory into the Docker container
COPY app /openvino/app

# Set the working directory in the container
WORKDIR /openvino/app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CACHE_DIR='/openvino/app/cache'
ENV DOCUMENT_DIR='/openvino/app/docs'
ENV CHROMA_PATH='/openvino/app/docs_embedding'

# Expose the ports where FastAPI and Streamlit servers will run
EXPOSE 8044 8504

# Run the run.py script to start both servers
CMD ["python", "run.py"]
