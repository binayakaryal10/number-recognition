"""
Preprocessing Utilities for Real-Time Digit Recognition
Prepares real-world camera crops to match the MNIST dataset distribution:
- Adaptive thresholding & noise removal
- Inversion (dark ink on white paper -> white stroke on black canvas)
- Aspect-ratio preserving resize into 20x20 box
- Center-of-mass centering on 28x28 canvas (exact MNIST standard)
"""

import cv2
import numpy as np

def shift_by_center_of_mass(img):
    """
    Shifts a 28x28 binary image so its center of mass is at (14, 14).
    Matches Yann LeCun's original MNIST centering normalization.
    """
    moments = cv2.moments(img)
    if moments['m00'] > 0:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        shift_x = np.round(14.0 - cx)
        shift_y = np.round(14.0 - cy)
        
        # Clamp shift to avoid pushing digit out of canvas
        shift_x = max(-5, min(5, shift_x))
        shift_y = max(-5, min(5, shift_y))
        
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(img, M, (28, 28), borderValue=0)
        return shifted
    return img

def preprocess_digit(roi_img, thicken_strokes=True):
    """
    Preprocess a BGR or Grayscale crop containing a handwritten digit into a
    normalized 28x28 array suitable for CNN prediction.
    
    Returns:
        input_tensor: np.ndarray of shape (1, 28, 28, 1), float32 in [0, 1]
        display_img: np.ndarray of shape (28, 28), uint8 (0-255) for UI preview
        is_empty: bool, True if no significant digit stroke was detected
    """
    if roi_img is None or roi_img.size == 0:
        empty = np.zeros((28, 28), dtype=np.uint8)
        return empty.reshape(1, 28, 28, 1).astype('float32'), empty, True

    # 1. Convert to grayscale if necessary
    if len(roi_img.shape) == 3:
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_img.copy()

    # 2. Denoise with Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Adaptive Inverted Thresholding
    # Standard paper is brighter than ink (mean > 120), so we invert: ink becomes white (255)
    mean_val = np.mean(blurred)
    if mean_val > 110:
        # Dark writing on light background
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 19, 9
        )
    else:
        # Light writing on dark background
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 19, 9
        )

    # Clean up minor speckles with opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 4. Find contours to locate the digit
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter noise: find the largest contour with sufficient area
    digit_contour = None
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area and area > 20:
            max_area = area
            digit_contour = c

    if digit_contour is None or max_area < 30:
        # No digit detected in this ROI
        empty = np.zeros((28, 28), dtype=np.uint8)
        return empty.reshape(1, 28, 28, 1).astype('float32'), empty, True

    # 5. Extract bounding box of the digit
    x, y, w, h = cv2.boundingRect(digit_contour)
    digit_crop = thresh[y:y+h, x:x+w]

    # Optionally thicken thin strokes (e.g. ballpoint pen ink)
    if thicken_strokes:
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        digit_crop = cv2.dilate(digit_crop, dilation_kernel, iterations=1)

    # 6. Aspect-ratio preserving resize into a 20x20 bounding box
    rows, cols = digit_crop.shape
    if rows > cols:
        factor = 20.0 / rows
        new_rows = 20
        new_cols = max(1, int(round(cols * factor)))
    else:
        factor = 20.0 / cols
        new_cols = 20
        new_rows = max(1, int(round(rows * factor)))

    resized = cv2.resize(digit_crop, (new_cols, new_rows), interpolation=cv2.INTER_AREA)

    # 7. Embed into a 28x28 black canvas with symmetric padding
    canvas = np.zeros((28, 28), dtype=np.uint8)
    pad_top = int(np.floor((28 - new_rows) / 2.0))
    pad_left = int(np.floor((28 - new_cols) / 2.0))
    canvas[pad_top:pad_top+new_rows, pad_left:pad_left+new_cols] = resized

    # 8. Shift by Center of Mass (MNIST standard)
    centered = shift_by_center_of_mass(canvas)

    # 9. Normalize to [0, 1] float32 for model input
    input_tensor = (centered.astype('float32') / 255.0).reshape(1, 28, 28, 1)

    return input_tensor, centered, False

def find_digit_regions(frame, min_area=300, max_area=50000):
    """
    Detects potential digit bounding boxes in the full frame for Auto-Contour mode.
    Returns list of (x, y, w, h) bounding boxes sorted left-to-right.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu inverse threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h
            # Digits typically have aspect ratio between 0.15 and 1.8
            if 0.15 < aspect_ratio < 2.0:
                boxes.append((x, y, w, h))
                
    # Sort boxes from left to right
    boxes.sort(key=lambda b: b[0])
    return boxes, thresh
