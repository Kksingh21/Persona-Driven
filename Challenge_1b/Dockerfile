# Adobe Hackathon Challenge 1B: Persona-Driven PDF Analysis
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all necessary application files
COPY . .

# Create mount point for collections
RUN mkdir -p /app/collections

# Set the entry point
CMD ["python", "analyze_collections.py"]
