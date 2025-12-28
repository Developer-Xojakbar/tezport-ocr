# Используем базовый образ с CUDA для поддержки GPU
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Устанавливаем переменные окружения
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Создаем символическую ссылку для python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Обновляем pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Устанавливаем PyTorch с CUDA поддержкой (для YOLO)
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Устанавливаем PaddlePaddle GPU версию
RUN pip3 install --no-cache-dir paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Устанавливаем остальные зависимости
RUN pip3 install --no-cache-dir \
    PaddleOCR==3.3.2 \
    Pillow==12.0.0 \
    numpy==2.4.0 \
    fastapi==0.127.0 \
    uvicorn==0.40.0 \
    python-multipart==0.0.21 \
    "ultralytics>=8.0.0"

# Копируем весь проект
COPY . .

# Создаем директорию для выходных файлов
RUN mkdir -p /app/output

# Открываем порт
EXPOSE 8080

# Запускаем приложение
CMD ["python", "main.py"]

