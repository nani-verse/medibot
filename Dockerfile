# Use Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch CPU version directly from PyTorch wheels
RUN pip install torch==2.2.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Copy requirements and install other Python dependencies
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copy app code
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "medibot.py", "--server.port=8501", "--server.address=0.0.0.0"]
