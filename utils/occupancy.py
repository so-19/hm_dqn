import numpy as np
import cv2

def compute_local_features(rgb: np.ndarray, bins: int = 8) -> np.ndarray:
    """
    Lightweight local perception vector (occupancy-ish).
    We use a horizontal edge histogram as a compact descriptor.
    """
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    h, w = edges.shape
    cols = np.array_split(edges, bins, axis=1)
    feat = np.array([float(np.mean(np.abs(c))) for c in cols], dtype=np.float32)
    feat = (feat - feat.mean()) / (feat.std() + 1e-6)
    return feat
