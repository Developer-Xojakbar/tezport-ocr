import io
from pathlib import Path
from typing import Union

from PIL import Image


def image_to_compress(
    image_path: Union[str, Path],
    target_size_kb: int = 20,
    quality: int = 85,
    log_size: bool = False,
) -> io.BytesIO:
    image_path = Path(image_path)
    target_size_bytes = target_size_kb * 1024
    

    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        current_size = target_size_bytes + 1
        current_quality = quality
        scale_factor = 1.0

        img_resized = img.copy()
        buffer = None
        
        while current_size > target_size_bytes and scale_factor > 0.3:
            if current_quality > 60:
                current_quality -= 5
            else:
                scale_factor -= 0.1
                new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
                img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            img_resized.save(
                buffer,
                format="JPEG",
                quality=current_quality,
                optimize=True,
            )
            current_size = buffer.tell()

            if current_size <= target_size_bytes:
                buffer.seek(0)
                break

        if buffer is None or current_size > target_size_bytes:
            if buffer is None:
                buffer = io.BytesIO()
                img_resized.save(
                    buffer,
                    format="JPEG",
                    quality=current_quality,
                    optimize=True,
                )
            buffer.seek(0)

    if log_size:
        initial_size = image_path.stat().st_size if log_size else 0 
        initial_size_kb = initial_size / 1024
        compressed_size = len(buffer.getvalue())
        compressed_size_kb = compressed_size / 1024

        print(f"Начальный размер: {initial_size_kb:.2f} KB ({initial_size} bytes)")
        print(f"Размер после сжатия: {compressed_size_kb:.2f} KB ({compressed_size} bytes)")

    return buffer

