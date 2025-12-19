#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_core.py: The core AI engine.
Contains logic for Feature Extraction, Anomaly Detection, Active Learning,
and Physics-Based (FFT) Validation.
"""
import torch
import timm
import numpy as np
import streamlit as st
import umap
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from utils import build_scan_map

# --- Helper for Patch Extraction ---
def extract_patches_single_size(img, patch_size, stride):
    patches, coords = [], []
    H, W = img.shape
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patches.append(img[i:i+patch_size, j:j+patch_size])
            coords.append((i, j, patch_size))
    return patches, coords

# --- Model Loading ---
@st.cache_resource
def load_clip_model():
    """Loads CLIP model for text-to-image search."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

@torch.no_grad()
def get_features(patches, backbone_name, device, batch_size=32):
    """Runs patches through a Vision Transformer or CNN."""
    if not patches: return np.empty((0, 0))

    try:
        model = timm.create_model(backbone_name, pretrained=True, num_classes=0)
    except Exception as e:
        st.error(f"Error loading {backbone_name}: {e}")
        return np.empty((0, 0))

    model = model.to(device).eval()
    
    feats = []
    for i in range(0, len(patches), batch_size):
        batch_list = patches[i:i+batch_size]
        batch_np = np.stack(batch_list, axis=0)
        
        x = torch.tensor(batch_np, dtype=torch.float32, device=device).unsqueeze(1)
        x = x.repeat(1, 3, 1, 1) 
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        output = model(x)
        f = output.detach().cpu().numpy()
        feats.append(f)
        
    return np.concatenate(feats, axis=0) if feats else np.empty((0,0))

# --- NEW: Physics-Based FFT Analysis ---
def compute_fft_metrics(patch):
    """
    Computes the Fast Fourier Transform (FFT) of a patch.
    Returns the log-magnitude spectrum and a 'Crystallinity Score'.
    """
    # 1. Apply Windowing (Hanning) to reduce edge artifacts
    h, w = patch.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    patch_windowed = patch * window
    
    # 2. Compute 2D FFT
    f_transform = np.fft.fft2(patch_windowed)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-9)
    
    # 3. Calculate Crystallinity (Variance of spectrum)
    # High variance = sharp diffraction spots (Crystalline)
    # Low variance = amorphous/blurry
    crystallinity_score = np.var(magnitude_spectrum)
    
    return magnitude_spectrum, crystallinity_score

# --- Main Pipelines ---
def run_analysis_pipeline(img, backbone, patch_sizes, strides, pca_dim=50):
    """Orchestrates the Unsupervised Anomaly Detection workflow."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_features, all_coords, all_patches_ref = [], [], []
    
    for ps, st in zip(patch_sizes, strides):
        p_curr, c_curr = extract_patches_single_size(img, ps, st)
        if not p_curr: continue
        f_curr = get_features(p_curr, backbone, device)
        all_features.append(f_curr)
        all_coords.extend(c_curr)
        all_patches_ref.extend(p_curr)
    
    if not all_features: return None

    features = np.concatenate(all_features, axis=0)
    
    # Dimensionality Reduction
    n_samples, n_dim = features.shape
    pca_n = min(pca_dim, n_samples, n_dim)
    features_reduced = PCA(n_components=pca_n).fit_transform(features) if pca_n > 1 else features
    
    # Anomaly Detection
    iso = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    iso.fit(features_reduced)
    raw_scores = -1 * iso.decision_function(features_reduced)
    mn, mx = raw_scores.min(), raw_scores.max()
    score = (raw_scores - mn) / (mx - mn) if mx > mn else np.zeros_like(raw_scores)
    
    # Visualization Embedding (UMAP)
    try:
        embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2).fit_transform(features_reduced)
    except:
        embedding = features_reduced[:, :2]

    return {
        "features": features_reduced,
        "coords": all_coords,
        "raw_patches": all_patches_ref,
        "score": score,
        "scan_map": build_scan_map(img.shape, all_coords, score),
        "embedding": embedding,
        "device": str(device)
    }

def train_classifier(features, coords, img_shape, pos_idx):
    """Active Learning: Trains a classifier based on user selection."""
    if pos_idx >= len(features): return np.zeros(len(features)), np.zeros(img_shape)

    X_train = [features[pos_idx]]
    y_train = [1]
    
    import random
    candidates = list(range(len(features)))
    candidates.remove(pos_idx)
    neg_indices = random.sample(candidates, min(5, len(candidates)))
    
    for idx in neg_indices:
        X_train.append(features[idx])
        y_train.append(0)
        
    clf = LogisticRegression(class_weight='balanced', C=10.0, solver='lbfgs')
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(features)[:, 1]
    
    mn, mx = probs.min(), probs.max()
    probs = (probs - mn) / (mx - mn) if mx > mn else np.zeros_like(probs)
    
    return probs, build_scan_map(img_shape, coords, probs)

@torch.no_grad()
def search_by_text(patches, text_query, img_shape, coords, device="cpu", batch_size=32):
    """Performs Semantic Search using CLIP."""
    model, processor = load_clip_model()
    model = model.to(device)
    
    inputs_text = processor(text=[text_query], return_tensors="pt", padding=True)
    text_features = model.get_text_features(input_ids=inputs_text["input_ids"].to(device),
                                            attention_mask=inputs_text["attention_mask"].to(device))
    text_features /= text_features.norm(dim=-1, keepdim=True) 

    scores = []
    for i in range(0, len(patches), batch_size):
        batch_list = patches[i:i+batch_size]
        batch_pil = [Image.fromarray((p * 255).astype(np.uint8)).convert("RGB") for p in batch_list]
        
        inputs_img = processor(images=batch_pil, return_tensors="pt")
        img_features = model.get_image_features(pixel_values=inputs_img["pixel_values"].to(device))
        img_features /= img_features.norm(dim=-1, keepdim=True)
        
        batch_scores = (img_features @ text_features.T).squeeze(1).cpu().numpy()
        scores.append(batch_scores)

    raw_scores = np.concatenate(scores)
    mn, mx = raw_scores.min(), raw_scores.max()
    text_scores = (raw_scores - mn) / (mx - mn) if mx > mn else np.zeros_like(raw_scores)
    
    text_map = build_scan_map(img_shape, coords, text_scores)
    return text_scores, text_map
