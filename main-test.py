import time
from pathlib import Path

from image_to_compress import image_to_compress
from image_to_text import image_to_text


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    test_dir = base_dir / "test"
    test_image = test_dir / "little_rotated.png"
    
    if test_image.exists():
        compress_start = time.time()
        compressed_image = image_to_compress(test_image, target_size_kb=20, log_size=True)
        compress_time = time.time() - compress_start
        print(f"Время сжатия: {compress_time:.2f} сек")
        
        ocr_start = time.time()
        result = image_to_text(compressed_image, min_score=0.9)
        ocr_time = time.time() - ocr_start
        print(f"Время OCR: {ocr_time:.2f} сек")
        
        data = result['data']
        texts = result['texts']

        print(f"rec_texts: {data['rec_texts']}")
        print(f"rec_scores: {data['rec_scores']}")
        print(f"texts: {texts}")
        print(f"Общее время: {compress_time + ocr_time:.2f} сек")
    else:
        print(f"Файл {test_image} не найден")


if __name__ == "__main__":
    main()


