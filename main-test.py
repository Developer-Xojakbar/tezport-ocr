import time
from pathlib import Path

from src.image_to_crop import image_to_crop
from src.image_to_compress import image_to_compress
from src.image_to_text import image_to_text
from src.get_info import get_info


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    test_dir = base_dir / "test"
    ocr_time_total = 0
    ocr_time_mean = 0
    ocr_success_count = 0
    ocr_scores_mean = 0
    compress_time_mean = 0
    crop_time_mean = 0
    
    files = ['TEMU6090861','LHXU7009100','TDRU5059997','WEDU8703933','GESU3684365','IMTU9038446','CAIU4032380','FCIU9332372','WSCU9579646','MSKU8074094','CCLU3834837','TLNU9101464','XINU1235818','PCHU9115162']
    # files = ['car1', 'car2', 'car3', 'car4', 'car5', 'car6']

    for base_name in files:
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
        crop_result = image_to_crop(test_image)
        detect = crop_result['detect']
        cropped_image = crop_result['image']
        crop_time = time.time() - crop_start
        crop_time_mean += crop_time

        compress_start = time.time()
        compressed_image = image_to_compress(cropped_image)
        compress_time = time.time() - compress_start
        compress_time_mean += compress_time
        
        ocr_start = time.time()
        result = image_to_text(compressed_image, detect=detect, save_to_output=True, output_name=base_name, print_variants=True)
        info = get_info(result['texts'], detect=detect, scores=result['data']['rec_scores'])
        ocr_time = time.time() - ocr_start
        ocr_time_mean += ocr_time
        
        
        texts = result['texts']

        ocr_scores_mean += info['scores_mean']
        print(f"texts: {info['scores_mean']}% - {texts}")
        ocr_time_total += crop_time + compress_time + ocr_time
       
        text = info.get('number') if info.get('number') else info.get('car')
        check_icon_text = "✅" if text == base_name else "❌"
        ocr_success_count += 1 if text == base_name else 0
        print(f"{check_icon_text}: {base_name} - {text}")
    
    crop_time_mean = crop_time_mean / len(files)
    compress_time_mean = compress_time_mean / len(files)
    ocr_time_mean = ocr_time_mean / len(files)
    crop_time_mean = round(crop_time_mean, 2)
    compress_time_mean = round(compress_time_mean, 2)
    ocr_time_mean = round(ocr_time_mean, 2)
    ocr_scores_mean = round(ocr_scores_mean / len(files), 2)

    print(f"Среднее время OCR: {ocr_time_mean} сек")
    print(f"Среднее время обрезки: {crop_time_mean} сек")
    print(f"Среднее время сжатия: {compress_time_mean} сек")
    print(f"Средний процент успешных OCR: {ocr_scores_mean}%")
    print(f"Успешных OCR: {ocr_success_count}/{len(files)}")
    print(f"Общее время: {ocr_time_total:.2f} сек")

if __name__ == "__main__":
    main()