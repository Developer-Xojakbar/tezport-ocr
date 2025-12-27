import time
from pathlib import Path

from src.image_to_compress import image_to_compress
from src.image_to_text import image_to_text
from src.get_container_info import get_container_info


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    test_dir = base_dir / "test"
    
    base_name = "MSKU2808138_NOT_VERTICAL"
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
    
    compress_start = time.time()
    compressed_image = image_to_compress(test_image, target_size_kb=40, log_size=True)
    compress_time = time.time() - compress_start
    print(f"Время сжатия: {compress_time:.2f} сек")
    
    ocr_start = time.time()
    result = image_to_text(compressed_image)
    container_info = get_container_info(result['texts'])
    ocr_time = time.time() - ocr_start
    print(f"Время OCR: {ocr_time:.2f} сек")
    
    data = result['data']
    texts = result['texts']

    print(f"rec_texts: {data['rec_texts']}")
    print(f"rec_scores: {data['rec_scores']}")
    print(f"texts: {texts}")
    print(f"container_info: {container_info}")
    print(f"Общее время: {compress_time + ocr_time:.2f} сек")


if __name__ == "__main__":
    main()


