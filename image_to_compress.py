import io
import tempfile
from pathlib import Path
from typing import Union

from PIL import Image


def image_to_compress(
    image_path: Union[str, Path],
    target_size_kb: int = 20,
    quality: int = 85,
) -> Path:
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

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg",
            dir=image_path.parent,
        )
        temp_path = Path(temp_file.name)

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
                with open(temp_path, "wb") as f:
                    f.write(buffer.read())
                break

        if buffer and current_size > target_size_bytes:
            buffer.seek(0)
            with open(temp_path, "wb") as f:
                f.write(buffer.read())

    return temp_path

