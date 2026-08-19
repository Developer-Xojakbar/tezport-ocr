# Linux/Vast.ai: CUDA 11.8, один Python-процесс (YOLO + PaddleOCR).
# На Linux cuDNN-конфликт Windows не воспроизводится, отдельный .venv-ocr не нужен.
# Python 3.12 — как локально; numpy 2.3.5 требует >=3.11.
# Torch 2.10.0+cu128 на CUDA 11.8 недоступен, поэтому cu118: torch 2.7.1.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES=0 \
    PORT=8080 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg \
    ca-certificates \
    wget \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && wget -q https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py \
    && python /tmp/get-pip.py \
    && rm /tmp/get-pip.py \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        torch==2.7.1 \
        torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu118 \
    && grep -vE '^paddlepaddle-gpu' requirements.txt > /tmp/requirements-docker.txt \
    && python -m pip install --no-cache-dir -r /tmp/requirements-docker.txt \
    && python -m pip install --no-cache-dir paddlepaddle-gpu==3.3.1 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ \
    && rm /tmp/requirements-docker.txt

COPY . .

RUN mkdir -p /app/output

EXPOSE 8080

CMD ["python", "main.py"]
