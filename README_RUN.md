```bash
# MACOS - Packages create + activate
python3 -m venv .venv
source .venv/bin/activate 

# Packages install
pip install -r requirements.txt


# WINDOWS - Packages create + activate
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Packages install
python -m pip install --upgrade pip
python -c "import platform; print(platform.system()); print(platform.machine())"

# PyTorch GPU (YOLO) — CUDA 12.8 для RTX 50xx
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

# PaddleOCR GPU — отдельное venv без PyTorch (конфликт cuDNN на Windows)
py -3.12 -m venv .venv-ocr
.\.venv-ocr\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
python -m pip install -r requirements-ocr.txt
deactivate

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Run Project
python main.py
```

На Windows YOLO и PaddleOCR оба на GPU: YOLO в `.venv`, PaddleOCR в `.venv-ocr` (CUDA 12.9, без PyTorch в том же процессе).
