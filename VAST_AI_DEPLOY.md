# Инструкция по деплою на Vast.ai

## ⚠️ ВАЖНО: Настройка SSH ключа (обязательно перед созданием инстанса)

Vast.ai требует SSH ключ для доступа к виртуальным машинам. Если вы получили ошибку `no_ssh_key_for_vm`, выполните следующие шаги:

### 1. Создание SSH ключа (если его нет)

```bash
# Проверьте, есть ли у вас SSH ключ
ls -la ~/.ssh/id_*.pub

# Если ключа нет, создайте его:
ssh-keygen -t ed25519 -C "your_email@example.com"

# Или используйте RSA (если ed25519 не поддерживается):
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# Нажмите Enter для всех вопросов (или укажите пароль для безопасности)
```

### 2. Получение публичного SSH ключа

```bash
# Покажите содержимое публичного ключа:
cat ~/.ssh/id_ed25519.pub

# Или если использовали RSA:
cat ~/.ssh/id_rsa.pub
```

Скопируйте весь вывод (начинается с `ssh-ed25519` или `ssh-rsa` и заканчивается вашим email).

### 3. Добавление SSH ключа в Vast.ai

1. Перейдите на страницу настроек: https://cloud.vast.ai/settings/
2. Найдите раздел **"SSH Keys"** или **"SSH Public Keys"**
3. Нажмите **"Add SSH Key"** или **"Add Key"**
4. Вставьте скопированный публичный ключ
5. Сохраните изменения

### 4. Проверка

После добавления ключа попробуйте снова создать инстанс. Ошибка `no_ssh_key_for_vm` больше не должна появляться.

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

#### Выбор шаблона (Template)

- **Используйте: Ubuntu 22.04 VM** (НЕ Ubuntu Desktop)
- Ubuntu Desktop не нужен, так как это веб-API без графического интерфейса
- Ubuntu 22.04 VM уже имеет Docker и NVIDIA Container Toolkit предустановлены

#### Настройки инстанса

- **Template**: Ubuntu 22.04 VM
- **GPU**: Выберите GPU с поддержкой CUDA (рекомендуется минимум 4GB VRAM, RTX 2060 Super отлично подходит)
- **Disk Size**: **40 GB** (рекомендуется) или минимум 30 GB
  - Docker образ: ~7-10 GB
  - Система и зависимости: ~10-15 GB
  - Запас для логов и обновлений: ~10-15 GB
- **Порт**: Откройте порт 8080 в настройках инстанса

#### Команда запуска в Vast.ai

После подключения к инстансу выполните:

**Если образ уже в Docker Hub:**

```bash
docker run --gpus all -p 8080:8080 -e PORT=8080 your-dockerhub-username/tezport-ocr:latest
```

**Если нужно собрать образ на инстансе:**

```bash
git clone https://github.com/your-username/tezport-ocr.git
cd tezport-ocr
docker build -t tezport-ocr .
docker run --gpus all -p 8080:8080 -e PORT=8080 tezport-ocr
```

**Для запуска в фоне (detached mode):**

```bash
docker run -d --gpus all -p 8080:8080 -e PORT=8080 --name tezport-ocr --restart unless-stopped your-image-name
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
