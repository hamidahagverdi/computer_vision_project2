import os
import cv2
import numpy as np

from utils import ensure_dir, list_images, imread_bgr, save_image, hstack_resize, normalize_to_uint8
from color_metrics import mse, psnr
from quantization import quantize_uniform, quantize_kmeans
from adjustments import adjust_hsv, adjust_hls
from deltae2000 import deltaE_ciede2000

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_DIR = os.path.join(ROOT, "images", "original")
OUT_DIR = os.path.join(ROOT, "outputs")

def grayscale_compare(img_bgr: np.ndarray):
    b = img_bgr[..., 0].astype(np.float32)
    g = img_bgr[..., 1].astype(np.float32)
    r = img_bgr[..., 2].astype(np.float32)

    # Proper luminance coefficients (sRGB/Rec.601 commonly used)
    gray_coeff = 0.114*b + 0.587*g + 0.299*r
    gray_avg = (b + g + r) / 3.0

    gc = normalize_to_uint8(gray_coeff)
    ga = normalize_to_uint8(gray_avg)

    diff = cv2.absdiff(gc, ga)

    metrics = {
        "mse": mse(gc, ga),
        "psnr": psnr(gc, ga),
    }

    return gc, ga, diff, metrics

def process_all_images():
    ensure_dir(OUT_DIR)

    imgs = list_images(IN_DIR)
    if not imgs:
        print(f"No images found in: {IN_DIR}")
        print("Put your photos into images/original/ and run again.")
        return

    for path in imgs:
        name = os.path.splitext(os.path.basename(path))[0]
        img = imread_bgr(path)

        # 1) Grayscale compare
        gc, ga, diff, metrics = grayscale_compare(img)
        save_image(os.path.join(OUT_DIR, "grayscale", f"{name}_gray_coeff.png"), gc)
        save_image(os.path.join(OUT_DIR, "grayscale", f"{name}_gray_avg.png"), ga)
        save_image(os.path.join(OUT_DIR, "grayscale", f"{name}_diff.png"), diff)

        # montage for qualitative comparison
        img_small = cv2.resize(img, (0,0), fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
        gc_bgr = cv2.cvtColor(gc, cv2.COLOR_GRAY2BGR)
        ga_bgr = cv2.cvtColor(ga, cv2.COLOR_GRAY2BGR)
        diff_bgr = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        montage = hstack_resize([img_small, gc_bgr, ga_bgr, diff_bgr], target_h=420)
        cv2.putText(montage, f"MSE={metrics['mse']:.2f}  PSNR={metrics['psnr']:.2f}dB",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        save_image(os.path.join(OUT_DIR, "grayscale", f"{name}_montage.png"), montage)

        # 2) Color quantization
        q_uni = quantize_uniform(img, levels_per_channel=4)
        q_k8 = quantize_kmeans(img, k=8)
        q_k16 = quantize_kmeans(img, k=16)

        save_image(os.path.join(OUT_DIR, "quantization", f"{name}_uniform4.png"), q_uni)
        save_image(os.path.join(OUT_DIR, "quantization", f"{name}_kmeans8.png"), q_k8)
        save_image(os.path.join(OUT_DIR, "quantization", f"{name}_kmeans16.png"), q_k16)

        # 3) Hue/Saturation/Brightness/Lightness changes (keep in valid ranges)
        hsv_mod = adjust_hsv(img, dh=15, ds=30, dv=25)     # hue shift + saturation + brightness
        hls_mod = adjust_hls(img, dh=-10, dl=25, ds=20)    # hue shift + lightness + saturation

        save_image(os.path.join(OUT_DIR, "hsl_hsv", f"{name}_hsv.png"), hsv_mod)
        save_image(os.path.join(OUT_DIR, "hsl_hsv", f"{name}_hls.png"), hls_mod)

        print(f"[OK] Processed: {os.path.basename(path)}")

def run_deltaE_picker(image_path: str, threshold: float = 10.0):
    img = imread_bgr(image_path)
    disp = img.copy()

    # Convert to Lab (OpenCV uses L in [0..255], a/b in [0..255] with 128 offset)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)

    clicked = {"pt": None, "lab": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["pt"] = (x, y)
            clicked["lab"] = lab[y, x, :].copy()

            ref = clicked["lab"][None, None, :]  # shape (1,1,3)
            de = deltaE_ciede2000(lab, ref)      # shape (H,W)

            mask = (de <= threshold).astype(np.uint8) * 255

            # Highlight similar colors: overlay green on masked pixels
            overlay = img.copy()
            overlay[mask == 255] = (0, 255, 0)

            blended = cv2.addWeighted(img, 0.65, overlay, 0.35, 0)
            cv2.circle(blended, (x, y), 6, (0, 0, 255), 2)

            out_base = os.path.splitext(os.path.basename(image_path))[0]
            out_dir = os.path.join(OUT_DIR, "deltaE")
            ensure_dir(out_dir)

            save_image(os.path.join(out_dir, f"{out_base}_mask_thr{threshold:.1f}.png"), mask)
            save_image(os.path.join(out_dir, f"{out_base}_highlight_thr{threshold:.1f}.png"), blended)

            cv2.imshow("DeltaE Similar Colors (click)", blended)

    cv2.namedWindow("DeltaE Similar Colors (click)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("DeltaE Similar Colors (click)", on_mouse)
    cv2.imshow("DeltaE Similar Colors (click)", disp)

    print("Click on the image to pick a color. Press ESC to exit.")
    while True:
        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Batch outputs for tasks 1-3:
    process_all_images()

    # Interactive task 4 (DeltaE):
    # Pick ONE image to demo color picking (change filename if you want)
    demo_images = list_images(IN_DIR)
    if demo_images:
        run_deltaE_picker(demo_images[0], threshold=10.0)