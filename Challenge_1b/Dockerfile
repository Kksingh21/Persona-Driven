# Adobe Hackathon Challenge 1B: Persona-Driven PDF Analysis
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install NLTK resources required by Sumy
RUN python -m nltk.downloader punkt

# Copy all source code
COPY . .

# Create mount point for collections
RUN mkdir -p /app/collections

# Set default entrypoint
CMD ["python", "analyze_collections.py"]
