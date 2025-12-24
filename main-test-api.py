import io
import uvicorn

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.get_container_info import get_container_info
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


@app.post("/ocr")
async def ocr_image(image: UploadFile = File(...)):
    content = await image.read()
    buffer = io.BytesIO(content)

    compressed_buffer = image_to_compress(buffer, target_size_kb=20, log_size=False)
    result = image_to_text(compressed_buffer)
    texts = result.get("texts", [])
    container_info = get_container_info(texts)

    return container_info


if __name__ == "__main__":
    uvicorn.run("main-test-api:app", host="localhost", port=8080, reload=True)

