from pathlib import Path
from image_to_text import image_to_text

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    test_dir = base_dir / "test"
    test_image = test_dir / "container-1 Paddle OCR.png"
    
    if test_image.exists():
        result = image_to_text(test_image)
        data = result['data']
        texts = result['texts']

        print(f"rec_texts: {data['rec_texts']}")
        print(f"rec_scores: {data['rec_scores']}")
        print(f"texts: {texts}")
    else:
        print(f"Файл {test_image} не найден")


if __name__ == "__main__":
    main()


