import cv2
import numpy as np

def adjust_hsv(img_bgr: np.ndarray, dh=0, ds=0, dv=0) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
    h, s, v = cv2.split(hsv)

    # Hue in OpenCV HSV: [0..179]
    h = (h + dh) % 180
    s = np.clip(s + ds, 0, 255)
    v = np.clip(v + dv, 0, 255)

    out = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_HSV2BGR)

def adjust_hls(img_bgr: np.ndarray, dh=0, dl=0, ds=0) -> np.ndarray:
    # OpenCV uses HLS (Hue, Lightness, Saturation)
    hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS).astype(np.int32)
    h, l, s = cv2.split(hls)

    # Hue: [0..179]
    h = (h + dh) % 180
    l = np.clip(l + dl, 0, 255)   # Lightness
    s = np.clip(s + ds, 0, 255)   # Saturation

    out = cv2.merge([h, l, s]).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_HLS2BGR)