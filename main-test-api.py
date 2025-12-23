import io

from fastapi import FastAPI, File, UploadFile

from src.get_container_info import get_container_info
from src.image_to_compress import image_to_compress
from src.image_to_text import image_to_text


app = FastAPI(title="Tezport OCR API")


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
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)

