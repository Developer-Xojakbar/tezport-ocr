import io
import os
import uvicorn

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.get_info import get_info
from src.image_to_compress import image_to_compress
from src.image_to_text import image_to_text


app = FastAPI(title="Tezport OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Tezport OCR API is running"}

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
    container_info = get_info(texts, detect=detect)

    return container_info


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main-test-api:app", host="0.0.0.0", port=port, reload=False)

