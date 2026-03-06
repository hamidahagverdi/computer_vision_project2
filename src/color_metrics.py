import numpy as np

def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean((a - b) ** 2))

def psnr(a: np.ndarray, b: np.ndarray, max_val=255.0) -> float:
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return float(20.0 * np.log10(max_val) - 10.0 * np.log10(m))