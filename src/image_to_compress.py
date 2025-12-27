import io
from pathlib import Path
from typing import Union

from PIL import Image


def image_to_compress(
    image_source: Union[str, Path, io.BytesIO],
    target_size_kb: int = 40,
    quality: int = 85,
    log_size: bool = False,
) -> io.BytesIO:
    target_size_bytes = target_size_kb * 1024

    image_path: Path | None = None
    initial_size = None
    
    if isinstance(image_source, (str, Path)):
        image_path = Path(image_source)
        initial_size = image_path.stat().st_size
        img = Image.open(image_path)
    elif isinstance(image_source, io.BytesIO):
        current_pos = image_source.tell()
        image_source.seek(0)
        initial_size = len(image_source.getvalue())
        image_source.seek(current_pos)
        image_source.seek(0)
        img = Image.open(image_source)
    
    
    if initial_size is not None and initial_size <= target_size_bytes:
        if log_size:
            initial_size_kb = initial_size / 1024
            print(f"Начальный размер: {initial_size_kb:.2f} KB ({initial_size} bytes)")
            print(f"Изображение уже меньше целевого размера ({target_size_kb} KB), сжатие не требуется")
        
        if isinstance(image_source, io.BytesIO):
            image_source.seek(0)
            return image_source
        else:
            buffer = io.BytesIO()
            with open(image_path, 'rb') as f:
                buffer.write(f.read())
            buffer.seek(0)
            return buffer

    with img:
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
        if initial_size is None:
            if image_path is not None:
                initial_size = image_path.stat().st_size
            else:
                initial_size = len(buffer.getvalue())
        
        initial_size_kb = initial_size / 1024
        compressed_size = len(buffer.getvalue())
        compressed_size_kb = compressed_size / 1024

        print(f"Начальный размер: {initial_size_kb:.2f} KB ({initial_size} bytes)")
        print(f"Размер после сжатия: {compressed_size_kb:.2f} KB ({compressed_size} bytes)")

    return buffer

