# Vast.ai Деплой

## SSH Key

```bash
# Создать ключ
ssh-keygen -t ed25519 -C "your_email@example.com"
# Нажать Enter для всех вопросов

# Получить публичный ключ
cat ~/.ssh/id_ed25519.pub
```

Добавить ключ: https://cloud.vast.ai/settings/ → SSH Keys → Add Key

## Create Instance

- **Template**: Ubuntu 22.04 VM
- **GPU**: RTX 2060 Super или выше (минимум 4GB VRAM)
- **Disk**: 40 GB
- **Ports**: Добавить порт `62900` (TCP)
- **onStart Script** (опционально):

## Build через Git Clone

```bash
git clone https://github.com/your-username/tezport-ocr.git
cd tezport-ocr
docker build -t tezport-ocr .
```

## Fix NVIDIA Bug

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update
apt-get install -y nvidia-container-toolkit
systemctl restart docker
```

## Get Containers

```bash
# Все контейнеры
docker ps -a

# Только запущенные
docker ps
```

## Delete Containers

```bash
# Остановить и удалить
docker stop tezport-ocr
docker rm tezport-ocr

# Или удалить все остановленные
docker container prune
```

## Update Project

```bash
cd tezport-ocr
git pull origin master
docker build -t tezport-ocr .
```

## Docker Run

```bash
# Запуск в фоне
docker run -d --gpus all -p 62900:8080 -e PORT=8080 --name tezport-ocr --restart unless-stopped tezport-ocr

# Проверка
docker ps
docker port tezport-ocr
docker logs tezport-ocr

# Тест API
curl http://localhost:62900/
curl http://localhost:62900/test-speed
```

## URL для React

```
http://142.170.89.112:62900
```

(Замените IP на ваш IP из настроек инстанса)
