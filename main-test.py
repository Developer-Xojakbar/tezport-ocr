import time
from pathlib import Path

from src.image_to_crop import image_to_crop
from src.image_to_compress import image_to_compress
from src.image_to_text import image_to_text
from src.get_info import get_info


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    test_dir = base_dir / "test"
    
    base_name = "TWCU8009897_NOT"
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    test_image = None
    
    for ext in image_extensions:
        candidate = test_dir / f"{base_name}{ext}"
        if candidate.exists():
            test_image = candidate
            break
    
    if test_image is None:
        print(f"Файл {base_name} с расширениями {image_extensions} не найден в {test_dir}")
        return

    crop_start = time.time()
    crop_result = image_to_crop(test_image, save_to_output=False)
    detect = crop_result['detect']
    cropped_image = crop_result['image']
    crop_time = time.time() - crop_start
    print(f"Время обрезки: {crop_time:.2f} сек")

    compress_start = time.time()
    compressed_image = image_to_compress(cropped_image, target_size_kb=40, log_size=True)
    compress_time = time.time() - compress_start
    print(f"Время сжатия: {compress_time:.2f} сек")
    
    ocr_start = time.time()
    result = image_to_text(compressed_image)
    info = get_info(result['texts'], detect=detect)
    ocr_time = time.time() - ocr_start
    print(f"Время OCR: {ocr_time:.2f} сек")
    
    data = result['data']
    texts = result['texts']

    print(f"rec_texts: {data['rec_texts']}")
    print(f"rec_scores: {data['rec_scores']}")
    print(f"texts: {texts}")
    print(f"info: {info}")
    print(f"Общее время: {crop_time + compress_time + ocr_time:.2f} сек")


if __name__ == "__main__":
    main()


