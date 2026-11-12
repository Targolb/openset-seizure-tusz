# PyTorch 2.3.0 + CUDA 12.1 + mamba-ssm 2.2.2
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace/openset-seizure-tusz

RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -U pip "setuptools>=69.5.1" wheel packaging ninja && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir --no-build-isolation "mamba-ssm==2.2.2"

COPY . .
CMD ["bash"]
