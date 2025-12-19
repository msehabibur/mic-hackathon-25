#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py: Utility functions for image loading, normalization, and patch extraction.
"""
import os
import numpy as np
from PIL import Image
from skimage import data, util

def load_image_grayscale(file_or_path) -> np.ndarray:
    """Loads an image from path or bytes and converts to normalized grayscale."""
    try:
        img = Image.open(file_or_path).convert("L")
        img = np.asarray(img, dtype=np.float32)
        denom = (img.max() - img.min())
        if denom < 1e-12: return np.zeros_like(img, dtype=np.float32)
        return (img - img.min()) / denom
    except Exception as e:
        return np.zeros((256, 256), dtype=np.float32)

def get_default_image() -> np.ndarray:
    """Checks for 'STEM_example' in root, otherwise uses fallback."""
    possible_names = ["STEM_example.png", "STEM_example.jpg", "STEM_example.tif"]
    for fname in possible_names:
        if os.path.exists(fname):
            return load_image_grayscale(fname)
    return util.img_as_float(data.brick())

def normalize(x: np.ndarray) -> np.ndarray:
    """Normalizes an array to 0-1 range."""
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8: return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def build_scan_map(img_shape, coords, score):
    """Reconstructs the heatmap from patch scores."""
    scan_map = np.zeros(img_shape, dtype=np.float32)
    count = np.zeros(img_shape, dtype=np.float32)
    for (i, j, size), s in zip(coords, score):
        scan_map[i:i+size, j:j+size] += float(s)
        count[i:i+size, j:j+size] += 1.0
    return scan_map / np.maximum(count, 1.0)

def extract_patches_single_size(img, patch_size, stride):
    """Extracts sliding window patches of a specific size."""
    patches, coords = [], []
    H, W = img.shape
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patches.append(img[i:i+patch_size, j:j+patch_size])
            coords.append((i, j, patch_size))
    return patches, coords
