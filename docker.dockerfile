FROM pytorch/pytorch:2.4.0-cuda12.4.1-cudnn8-runtime

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    nano \
    unzip \
    htop \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --upgrade pip && pip install \
    jupyterlab \
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    seaborn \
    transformers \
    einops \
    timm \
    opencv-python \
    torchsummary \
    psutil \
    memory-profiler \
    tqdm \
    ipywidgets

# Expose JupyterLab port
EXPOSE 8888

# Start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
