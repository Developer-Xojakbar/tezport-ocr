import io
import os
import uvicorn

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.image_to_crop import image_to_crop
from src.get_info import get_info
from src.image_to_compress import image_to_compress
from src.image_to_text import image_to_text
from src.test_speed import test_speed


app = FastAPI(title="Tezport OCR API")

ALLOWED_ORIGINS = [
    'https://tezport-ui-dev.onrender.com',
    'https://tezport-ui-prod.onrender.com',
    'https://tezport-app-dev.onrender.com',
    'https://tezport-app-prod.onrender.com',
    'https://www.tezport.dev',
    'https://www.tezport.com',
    'https://www.tezport.app',
    'https://tezport.dev',
    'https://tezport.com',
    'https://tezport.app',
    'https://tezport-api-dev.onrender.com',
    'https://tezport-api-prod.onrender.com',
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Tezport OCR API is running"}

@app.get("/test-speed")
async def test_speed_local():
    return test_speed()

@app.post("/ocr")
async def ocr_image(image: UploadFile = File(...)):
    content = await image.read()
    buffer = io.BytesIO(content)

    crop_result = image_to_crop(buffer, save_to_output=False)
    detect = crop_result['detect']
    cropped_image = crop_result['image']

    compressed_buffer = image_to_compress(cropped_image)
    result = image_to_text(compressed_buffer)
    texts = result.get("texts", [])
    info = get_info(texts, detect=detect)

    return info


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

