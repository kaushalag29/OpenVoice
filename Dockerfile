# Base image with Anaconda
FROM continuumio/miniconda3

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DOCKER=true

# Set working directory
WORKDIR /app

# Install system dependencies
# git is needed for some pip installs from git repos
# ffmpeg is a common dependency for audio processing libraries
# build-essential and other build tools are needed for compiling packages
# unzip is needed for extracting downloaded checkpoint files
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    build-essential \
    gcc \
    g++ \
    unzip \
    wget \
    && apt-get clean

# Copy the application files into the container
COPY . .

# Create a conda environment for the application
# OpenVoice requires Python 3.9.23 according to TODO.md
RUN conda create -n openvoice python=3.9 -y

# Activate the conda environment for subsequent commands
SHELL ["conda", "run", "-n", "openvoice", "/bin/bash", "-c"]

# Upgrade pip to get latest wheel support
RUN pip install --upgrade pip

# Install dependencies from requirements.txt with timeout and retry
RUN pip install -r requirements.txt --timeout=100 --retries=3

# Install OpenVoice package in development mode
RUN pip install -e .

# Install MeloTTS (required for OpenVoice V2 as per USAGE.md)
RUN pip install git+https://github.com/myshell-ai/MeloTTS.git

# Download and prepare Unidic (required for MeloTTS as per USAGE.md)
RUN python -m unidic download

# Download and unzip OpenVoice checkpoint inside the OpenVoice directory
# This ensures checkpoints are in the correct location relative to server.py
# Use -o flag to overwrite existing files without prompting
RUN wget -O checkpoints_v2_0417.zip https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip && \
    unzip -o checkpoints_v2_0417.zip && \
    rm checkpoints_v2_0417.zip

# Expose the port the server will run on (port 8009 as specified in server.py)
EXPOSE 8009

# Command to run the application server with conda environment activated
CMD ["conda", "run", "-n", "openvoice", "python", "server.py"] 