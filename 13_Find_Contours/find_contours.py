import numpy as np
import cv2

font = cv2.FONT_HERSHEY_COMPLEX

# Load image
img2 = cv2.imread('src/111.png', cv2.IMREAD_COLOR)
img = cv2.imread('src/111.png', cv2.IMREAD_GRAYSCALE)

# Binarize image
_, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    # Approximate and draw contour
    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
    cv2.drawContours(img2, [approx], 0, (0, 0, 255), 1)

    # Flatten points
    n = approx.ravel()
    i = 0
    for j in n:
        if i % 2 == 0:  # x, y coords
            x, y = n[i], n[i + 1]
            coord = f"{x} {y}"
            if i == 0:  # first point
                cv2.putText(img2, "Arrow tip", (x, y), font, 0.5, (255, 0, 0))
            else:
                cv2.putText(img2, coord, (x, y), font, 0.5, (0, 255, 0))
        i += 1

# Show result
cv2.imshow('Contours with Coordinates', img2)

# Exit on 'q'
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()