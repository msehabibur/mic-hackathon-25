# al_core.py
import numpy as np
import torch
import timm
import umap
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance

torch.set_default_dtype(torch.float32)

def load_image_grayscale(file) -> np.ndarray:
    img = Image.open(file).convert("L")
    img = np.asarray(img, dtype=np.float32)
    denom = (img.max() - img.min())
    if denom < 1e-12:
        return np.zeros_like(img, dtype=np.float32)
    img = (img - img.min()) / denom
    return img

def extract_patches(img: np.ndarray, patch_size: int, stride: int):
    patches, coords = [], []
    H, W = img.shape
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patches.append(img[i:i+patch_size, j:j+patch_size])
            coords.append((i, j, patch_size))
    return patches, coords  # list is important

@torch.no_grad()
def encode_patches(patches, model, device, batch_size=64):
    feats = []
    for i in range(0, len(patches), batch_size):
        batch = np.stack(patches[i:i+batch_size], axis=0)  # (B,H,W)
        x = torch.tensor(batch, dtype=torch.float32, device=device).unsqueeze(1)  # (B,1,H,W)
        x = x.repeat(1, 3, 1, 1)  # (B,3,H,W)
        f = model(x).detach().cpu().numpy().astype(np.float32)
        feats.append(f)
    return np.concatenate(feats, axis=0)

def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + 1e-8)

def build_scan_map(img, coords, score):
    scan_map = np.zeros_like(img, dtype=np.float32)
    count = np.zeros_like(img, dtype=np.float32)
    for (i, j, size), s in zip(coords, score):
        scan_map[i:i+size, j:j+size] += float(s)
        count[i:i+size, j:j+size] += 1.0
    scan_map = scan_map / np.maximum(count, 1.0)
    return scan_map

def run_active_learning(
    img: np.ndarray,
    backbone="resnet18",
    patch_sizes=(32, 64),
    strides=(16, 32),
    top_k=10,
    pca_dim=50,
    umap_neighbors=15,
    umap_min_dist=0.1,
    batch_size=64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model(backbone, pretrained=True)
    model.reset_classifier(0)
    model = model.to(device).eval()

    all_features = []
    all_coords = []

    for ps, st in zip(patch_sizes, strides):
        patches, coords = extract_patches(img, ps, st)
        if len(patches) == 0:
            continue
        feats = encode_patches(patches, model, device, batch_size=batch_size)
        all_features.append(feats)
        all_coords.extend(coords)

    features = np.concatenate(all_features, axis=0).astype(np.float32)

    # --- Informativeness metrics ---
    pca_dim = int(min(pca_dim, features.shape[1], max(2, features.shape[0] - 1)))
    pca = PCA(n_components=pca_dim)
    Z = pca.fit_transform(features).astype(np.float32)

    cov = EmpiricalCovariance().fit(Z)
    novelty = cov.mahalanobis(Z).astype(np.float32)

    uncertainty = np.linalg.norm(features - features.mean(axis=0), axis=1).astype(np.float32)
    diversity = np.linalg.norm(features, axis=1).astype(np.float32)

    score = normalize(normalize(novelty) + normalize(uncertainty) + normalize(diversity))

    # scan map
    scan_map = build_scan_map(img, all_coords, score)

    # UMAP
    embedding = umap.UMAP(n_neighbors=umap_neighbors, min_dist=umap_min_dist).fit_transform(features)
    embedding = embedding.astype(np.float32)

    # Top-K
    top_k = int(min(top_k, len(score)))
    top_idx = np.argsort(score)[-top_k:][::-1]  # high→low

    top_regions = []
    for rank, idx in enumerate(top_idx, start=1):
        i, j, size = all_coords[idx]
        top_regions.append({
            "rank": rank,
            "i": int(i),
            "j": int(j),
            "size": int(size),
            "score": float(score[idx]),
        })

    return {
        "device": str(device),
        "features_shape": features.shape,
        "coords": all_coords,
        "score": score,
        "scan_map": scan_map,
        "embedding": embedding,
        "top_regions": top_regions,
    }
