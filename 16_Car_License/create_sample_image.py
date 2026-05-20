"""Create a demo image sized for PlateFinder (area 4100–15000, ratio ~3–6)."""
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "src" / "car.jpg"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    h_img, w_img = 480, 720
    img = np.full((h_img, w_img, 3), 55, dtype=np.uint8)

    # Car body gradient
    for y in range(h_img):
        img[y, :, :] = (40 + y // 8, 45 + y // 10, 50 + y // 12)

    # Plate: ~150x30 px => area 4500, aspect ratio 5
    pw, ph = 150, 30
    x, y = (w_img - pw) // 2, h_img - 120
    cv2.rectangle(img, (x - 2, y - 2), (x + pw + 2, y + ph + 2), (30, 30, 30), 2)
    cv2.rectangle(img, (x, y), (x + pw, y + ph), (235, 235, 235), -1)

    text = "29A33185"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    tx = x + (pw - tw) // 2
    ty = y + (ph + th) // 2
    cv2.putText(img, text, (tx, ty), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Slight blur on background only (plate stays sharp)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.rectangle(mask, (x, y), (x + pw, y + ph), 255, -1)
    img = np.where(mask[..., None] == 255, img, blurred)

    cv2.imwrite(str(OUT), img)
    print(f"Wrote {OUT} (plate area {pw * ph}, ratio {pw / ph:.1f})")


if __name__ == "__main__":
    main()
