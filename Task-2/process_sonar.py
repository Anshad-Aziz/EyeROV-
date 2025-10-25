import cv2
import numpy as np
from bm3d import bm3d, BM3DProfile
# Load as grayscale
img = cv2.imread("oculus.jpg", 0)

def lee_filter(img, size=7, cu=0.5):
    img = img.astype(np.float32)
    mean = cv2.blur(img, (size, size))
    mean_sq = cv2.blur(img**2, (size, size))
    var = mean_sq - mean**2

    eps = 1e-8
    w = 1 - (cu / (var / (mean**2 + eps) + cu))

    result = mean + w * (img - mean)
    result = np.nan_to_num(result)
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)


lee = lee_filter(img)
cv2.imwrite("output_lee.png", lee)

def kuan_filter(img, size=7):
    img = img.astype(np.float32)
    mean = cv2.blur(img, (size, size))
    mean_sq = cv2.blur(img**2, (size, size))
    var = mean_sq - mean**2

    cu = 0.25
    w = 1 - (cu / (var / (mean**2) + cu))
    return (mean + w * (img - mean)).astype(np.uint8)

kuan = kuan_filter(img)
cv2.imwrite("output_kuan.png", kuan)


def frost_filter(img, size=5, damping=1):
    img = img.astype(np.float32)
    rows, cols = img.shape
    result = np.zeros_like(img)

    pad = size // 2
    padded = np.pad(img, pad, mode='reflect')

    for i in range(rows):
        for j in range(cols):
            window = padded[i:i+size, j:j+size]
            mean = np.mean(window)
            var = np.var(window)
            coef = np.exp(-damping * np.abs(window - mean) / (var + 1e-5))
            result[i, j] = np.sum(window * coef) / np.sum(coef)

    return result.astype(np.uint8)

frost = frost_filter(img)
cv2.imwrite("output_frost.png", frost)

def srad(img, num_iter=80, delta=0.25):
    img = img.astype(np.float32) / 255.0
    for _ in range(num_iter):
        n = np.roll(img, -1, axis=0) - img
        s = np.roll(img, 1, axis=0) - img
        e = np.roll(img, -1, axis=1) - img
        w = np.roll(img, 1, axis=1) - img

        g2 = (n**2 + s**2 + e**2 + w**2)
        q2 = g2.mean()

        c = np.exp(-(g2 - q2) / (q2 + 1e-12))
        img += delta * (c*n + c*s + c*e + c*w)

    return (img * 255).astype(np.uint8)

srad_out = srad(img)
cv2.imwrite("output_srad.png", srad_out)


bm3d_out = bm3d(img, sigma_psd=15/255, profile=BM3DProfile())
cv2.imwrite("output_bm3d.png", (bm3d_out*255).astype(np.uint8))

clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))

enhanced_srad = clahe.apply(srad_out)
cv2.imwrite("output_srad_enhanced.png", enhanced_srad)
