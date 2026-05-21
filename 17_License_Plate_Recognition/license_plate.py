import shutil
import sys
from pathlib import Path

import cv2
import pytesseract

SCRIPT_DIR = Path(__file__).resolve().parent


def _configure_tesseract() -> None:
    """Point pytesseract at the Tesseract binary (Linux tutorial path vs Windows install)."""
    if sys.platform == "win32":
        for candidate in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ):
            if Path(candidate).is_file():
                pytesseract.pytesseract.tesseract_cmd = candidate
                return
        found = shutil.which("tesseract")
        if found:
            pytesseract.pytesseract.tesseract_cmd = found
            return
        raise FileNotFoundError(
            "Tesseract OCR not found. Install from "
            "https://github.com/UB-Mannheim/tesseract/wiki "
            "and ensure tesseract.exe is on PATH or in Program Files."
        )
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


def detect_plate_number(image_path: Path) -> str | None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    contours, _ = cv2.findContours(
        edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    plate_contour = None
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            plate_contour = approx
            break

    if plate_contour is None:
        return None

    x, y, w, h = cv2.boundingRect(plate_contour)
    plate_image = gray[y : y + h, x : x + w]
    _, thresh = cv2.threshold(
        plate_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    plate_number = pytesseract.image_to_string(thresh, config="--psm 8")
    return plate_number.strip()


def _default_image() -> Path:
    local = SCRIPT_DIR / "car1.jpg"
    if local.is_file():
        return local
    fallback = SCRIPT_DIR.parent / "16_Car_License" / "src" / "car.jpg"
    if fallback.is_file():
        return fallback
    return local


if __name__ == "__main__":
    _configure_tesseract()
    image_path = _default_image()
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Place a test image at {SCRIPT_DIR / 'car1.jpg'} "
            f"or ensure {SCRIPT_DIR.parent / '16_Car_License' / 'src' / 'car.jpg'} exists."
        )
    plate_number = detect_plate_number(image_path)
    if plate_number:
        print("Detected Plate Number:", plate_number)
    else:
        print("No license plate contour found in", image_path)
