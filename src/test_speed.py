import time
from pathlib import Path

from src.image_to_crop import image_to_crop
from src.image_to_compress import image_to_compress
from src.image_to_text import image_to_text
from src.get_info import get_info

def test_speed():
    project_root = Path(__file__).resolve().parent.parent
    test_dir = project_root / "test"
    ocr_time_total = 0
    ocr_time_mean = 0
    ocr_success_count = 0
    ocr_scores_mean = 0
    compress_time_mean = 0
    crop_time_mean = 0

    files = ['TEMU6090861','LHXU7009100','TDRU5059997','WEDU8703933','GESU3684365','IMTU9038446','CAIU4032380','FCIU9332372','WSCU9579646','MSKU8074094','CCLU3834837','TLNU9101464','XINU1235818','PCHU9115162']
    test_images = []
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    for base_name in files:
        for ext in image_extensions:
            candidate = test_dir / f"{base_name}{ext}"
            if candidate.exists():
                test_images.append(candidate)
                break

    if len(test_images) == 0:
        return None

    n = len(test_images)
    for test_image in test_images:
        expected = test_image.stem
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
        result = image_to_text(compressed_image, detect=detect)
        info = get_info(result['texts'], detect=detect, scores=result['data']['rec_scores'])
        ocr_time = time.time() - ocr_start
        ocr_time_mean += ocr_time
        
        
        ocr_scores_mean += info['scores_mean']
        ocr_time_total += crop_time + compress_time + ocr_time
        text = info.get('number') if info.get('number') else info.get('car')
        ocr_success_count += 1 if text == expected else 0

    crop_time_mean = crop_time_mean / n
    compress_time_mean = compress_time_mean / n
    ocr_time_mean = ocr_time_mean / n
    crop_time_mean = round(crop_time_mean, 2)
    compress_time_mean = round(compress_time_mean, 2)
    ocr_time_mean = round(ocr_time_mean, 2)
    ocr_scores_mean = round(ocr_scores_mean / n, 2)

    return {
        "Среднее время OCR": f"{ocr_time_mean} сек",
        "Среднее время обрезки": f"{crop_time_mean} сек",
        "Среднее время сжатия": f"{compress_time_mean} сек",
        "Средний процент успешных OCR": f"{ocr_scores_mean}%",
        "Успешных OCR": f"{ocr_success_count}/{n}",
        "Общее время": f"{ocr_time_total:.2f} сек",
    }