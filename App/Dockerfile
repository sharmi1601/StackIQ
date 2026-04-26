FROM python:3.10-slim

# Install system dependencies needed by faiss and torch
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements first (Docker caches this layer)
# So if only your code changes, it won't reinstall packages
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Pre-download the models during build
# So your friend doesn't wait for model downloads when container starts
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy your backend code
COPY backend.py .

# Start the API
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]