import os
import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def list_images(folder: str):
    files = []
    if not os.path.isdir(folder):
        return files
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(IMG_EXTS):
            files.append(os.path.join(folder, f))
    return files

def imread_bgr(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def save_image(path: str, img: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"Failed to write: {path}")

def hstack_resize(images, target_h=480):
    # Resize images to same height then hstack
    resized = []
    for im in images:
        h, w = im.shape[:2]
        scale = target_h / float(h)
        new_w = int(round(w * scale))
        resized.append(cv2.resize(im, (new_w, target_h), interpolation=cv2.INTER_AREA))
    return cv2.hconcat(resized)

def normalize_to_uint8(gray_float):
    g = np.clip(gray_float, 0, 255).astype(np.uint8)
    return g