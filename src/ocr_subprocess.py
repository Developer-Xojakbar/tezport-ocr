import base64
import io
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from src.ocr_config import PROJECT_ROOT

OCR_VENV_PYTHON = PROJECT_ROOT / ".venv-ocr" / "Scripts" / "python.exe"
if not OCR_VENV_PYTHON.exists():
    OCR_VENV_PYTHON = PROJECT_ROOT / ".venv-ocr" / "bin" / "python"

_client: Optional["OcrSubprocessClient"] = None


def _get_worker_python() -> str:
    if OCR_VENV_PYTHON.exists():
        return str(OCR_VENV_PYTHON)
    return sys.executable


class OcrSubprocessClient:
    def __init__(self) -> None:
        self._proc = subprocess.Popen(
            [_get_worker_python(), "-m", "src.ocr_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT),
        )
        self._lock = threading.Lock()
        self._gpu = False
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self._wait_ready()

    def _drain_stderr(self) -> None:
        if self._proc.stderr is None:
            return
        for line in self._proc.stderr:
            line = line.rstrip()
            if line:
                print(line, file=sys.stderr)

    def _wait_ready(self) -> None:
        if self._proc.stdout is None:
            raise RuntimeError("OCR worker stdout недоступен")

        ready_line = self._proc.stdout.readline()
        if not ready_line:
            raise RuntimeError("OCR worker завершился до инициализации")

        payload = json.loads(ready_line)
        if not payload.get("ready"):
            raise RuntimeError(f"OCR worker не готов: {payload}")

        self._gpu = bool(payload.get("gpu"))
        device = payload.get("device", "cpu")
        container = payload.get("container", {})
        car = payload.get("car", {})
        print(f"✅ PaddleOCR запущен в отдельном процессе ({device}).")
        print(f"   container: det={container.get('det')} rec={container.get('rec')}")
        print(f"   car: det={car.get('det')} rec={car.get('rec')}")

    def predict(self, version: str, image: np.ndarray, min_score: float) -> Dict[str, List]:
        with self._lock:
            if self._proc.poll() is not None:
                raise RuntimeError("OCR worker завершился неожиданно")

            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

            request = {
                "version": version,
                "min_score": min_score,
                "image_b64": image_b64,
            }
            assert self._proc.stdin is not None
            assert self._proc.stdout is not None

            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()

            response_line = self._proc.stdout.readline()
            if not response_line:
                raise RuntimeError("OCR worker не вернул ответ")

            payload = json.loads(response_line)
            if not payload.get("ok"):
                raise RuntimeError(payload.get("error", "OCR worker error"))

            return {
                "rec_texts": payload.get("rec_texts", []),
                "rec_scores": payload.get("rec_scores", []),
                "rec_bboxes": payload.get("rec_bboxes", []),
            }

    @property
    def gpu(self) -> bool:
        return self._gpu


def get_ocr_subprocess_client() -> OcrSubprocessClient:
    global _client
    if _client is None:
        _client = OcrSubprocessClient()
    return _client
