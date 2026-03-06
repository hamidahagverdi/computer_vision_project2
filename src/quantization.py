import cv2
import numpy as np

def quantize_uniform(img_bgr: np.ndarray, levels_per_channel: int = 4) -> np.ndarray:
    # levels_per_channel: 2..16 typical. 4 => 64 colors approx.
    levels = max(2, min(256, int(levels_per_channel)))
    step = 256 // levels
    out = (img_bgr // step) * step + step // 2
    return np.clip(out, 0, 255).astype(np.uint8)

def quantize_kmeans(img_bgr: np.ndarray, k: int = 8, attempts: int = 10) -> np.ndarray:
    Z = img_bgr.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
    _compactness, labels, centers = cv2.kmeans(
        Z, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )
    centers = np.uint8(centers)
    quant = centers[labels.flatten()]
    return quant.reshape(img_bgr.shape)