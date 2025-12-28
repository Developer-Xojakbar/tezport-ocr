# Инструкция по деплою на Vast.ai

## Подготовка Docker образа

### 1. Сборка образа локально (опционально)

```bash
docker build -t tezport-ocr:latest .
```

### 2. Тестирование образа локально (если есть GPU)

```bash
docker run --gpus all -p 8080:8080 tezport-ocr:latest
```

## Деплой на Vast.ai

### Вариант 1: Использование готового образа

1. Загрузите образ в Docker Hub:

```bash
docker tag tezport-ocr:latest your-dockerhub-username/tezport-ocr:latest
docker push your-dockerhub-username/tezport-ocr:latest
```

2. На Vast.ai используйте команду:

```bash
docker run --gpus all -p 8080:8080 your-dockerhub-username/tezport-ocr:latest
```

### Вариант 2: Сборка на Vast.ai

1. Загрузите проект на GitHub или другой репозиторий
2. На Vast.ai используйте команду:

```bash
git clone https://github.com/your-username/tezport-ocr.git
cd tezport-ocr
docker build -t tezport-ocr .
docker run --gpus all -p 8080:8080 tezport-ocr
```

### Настройки на Vast.ai

- **GPU**: Выберите GPU с поддержкой CUDA (рекомендуется минимум 4GB VRAM)
- **Порт**: Откройте порт 8080
- **Команда запуска**:
  ```bash
  docker run --gpus all -p 8080:8080 -e PORT=8080 your-image-name
  ```

### Проверка работы

После запуска проверьте:

```bash
curl http://localhost:8080/
```

Должен вернуться:

```json
{ "status": "ok", "message": "Tezport OCR API is running" }
```

### Тестирование OCR API

```bash
curl -X POST "http://your-vast-ai-ip:8080/ocr" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/path/to/test/image.jpg"
```

## Переменные окружения

- `PORT` - порт для запуска API (по умолчанию 8080)
- `CUDA_VISIBLE_DEVICES` - какие GPU использовать (по умолчанию 0)
- `YOLO_LICENSE_PLATE_MODEL` - путь к кастомной модели YOLO для номеров машин
- `YOLO_CONTAINER_MODEL` - путь к кастомной модели YOLO для контейнеров

## Требования к GPU

- Минимум 4GB VRAM
- CUDA 11.8 или выше
- cuDNN 8.0 или выше
