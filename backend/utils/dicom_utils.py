import pydicom
import numpy as np
import cv2


def load_dicom(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)

    # normalize
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def resize_square(img, size=256):
    h, w = img.shape
    s = size

    scale = min(s / h, s / w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((s, s), dtype=np.float32)

    y = (s - new_h) // 2
    x = (s - new_w) // 2
    canvas[y:y+new_h, x:x+new_w] = resized

    return canvas
