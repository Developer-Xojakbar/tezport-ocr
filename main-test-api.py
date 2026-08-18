import io
import json
import os
import uvicorn

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from src.image_to_crop import image_to_crop
from src.get_info import get_info
from src.image_to_text import image_to_text
from src.test_speed import test_speed


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
    data = {"status": "ok", "message": "Tezport OCR API is running"}
    return Response(
        content=json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        media_type="application/json; charset=utf-8",
    )

@app.get("/test-speed")
async def test_speed_local():
    data = test_speed() 
    return Response(
        content=json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        media_type="application/json; charset=utf-8",
    )

@app.post("/ocr")
async def ocr_image(image: UploadFile = File(...)):
    content = await image.read()
    buffer = io.BytesIO(content)

    crop_result = image_to_crop(buffer)
    detect = crop_result['detect']
    cropped_image = crop_result['image']

    result = image_to_text(cropped_image, detect=detect)
    texts = result.get("texts", [])
    info = get_info(texts, detect=detect)

    return info


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))
    uvicorn.run("main-test-api:app", host="0.0.0.0", port=port, reload=False)

