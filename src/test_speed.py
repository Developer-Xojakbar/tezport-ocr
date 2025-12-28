import time
from pathlib import Path

from src.image_to_crop import image_to_crop
from src.image_to_compress import image_to_compress
from src.image_to_text import image_to_text
from src.get_info import get_info

def test_speed():
    base_dir = Path(__file__).resolve().parent
    test_dir = base_dir
    
    base_name = "MSKU8074094"
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    test_image = None
    
    for ext in image_extensions:
        candidate = test_dir / f"{base_name}{ext}"
        if candidate.exists():
            test_image = candidate
            break
    
    if test_image is None:
        return None

    crop_start = time.time()
    crop_result = image_to_crop(test_image)
    detect = crop_result['detect']
    cropped_image = crop_result['image']
    crop_time = time.time() - crop_start

    compress_start = time.time()
    compressed_image = image_to_compress(cropped_image, target_size_kb=40)
    compress_time = time.time() - compress_start
    
    ocr_start = time.time()
    result = image_to_text(compressed_image)
    info = get_info(result['texts'], detect=detect)
    ocr_time = time.time() - ocr_start
    
    data = result['data']
    texts = result['texts']


    return {
        'crop_time': f"Время обрезки: {crop_time:.2f} сек",
        'compress_time': f"Время сжатия: {compress_time:.2f} сек",
        'ocr_time': f"Время OCR: {ocr_time:.2f} сек",
        'rec_texts': f"rec_texts: {data['rec_texts']}",
        'rec_scores': f"rec_scores: {data['rec_scores']}",
        'texts': f"texts: {texts}",
        'info': f"info: {info}",
        'total_time': f"Общее время: {crop_time + compress_time + ocr_time:.2f} сек",
    }