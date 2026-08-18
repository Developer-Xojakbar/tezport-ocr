"""Глобальные настройки проекта. Меняйте только этот файл."""

# Папки внутри paddle_models/
#
# det:
#   PP-OCRv6_medium_det
#   PP-OCRv5_server_det
#
# rec:
#   container_rec_v6_infer          — ваша fine-tune v6
#   container_server_rec_infer_v5   — ваша fine-tune v5
#   container_rec_infer_v3          — ваша fine-tune v3
#   PP-OCRv6_medium_rec             — официальная v6
#   PP-OCRv5_server_rec             — официальная v5

CONTAINER_DET = "PP-OCRv6_medium_det"
CONTAINER_REC = "PP-OCRv6_medium_rec"

CAR_DET = "PP-OCRv6_medium_det"
CAR_REC = "PP-OCRv6_medium_rec"
