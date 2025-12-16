import numpy as np
import torch
import timm
import umap
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity

# Ensure consistent float type
torch.set_default_dtype(torch.float32)

def load_image_grayscale(file) -> np.ndarray:
    """Loads image, converts to grayscale, and normalizes to [0,1]."""
    try:
        img = Image.open(file).convert("L")
        img = np.asarray(img, dtype=np.float32)
        denom = (img.max() - img.min())
        if denom < 1e-12:
            return np.zeros_like(img, dtype=np.float32)
        img = (img - img.min()) / denom
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return np.zeros((256, 256), dtype=np.float32)

def extract_patches(img: np.ndarray, patch_size: int, stride: int):
    """Extracts patches and records their top-left coordinates (i, j, size)."""
    patches, coords = [], []
    H, W = img.shape
    # Ensure we don't error on small images
    if H < patch_size or W < patch_size:
        return [], []
        
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patches.append(img[i:i+patch_size, j:j+patch_size])
            coords.append((i, j, patch_size))
    return patches, coords

@torch.no_grad()
def encode_patches(patches, model, device, batch_size=64):
    """Passes patches through the deep learning backbone."""
    if not patches:
        return np.empty((0, 0))
        
    feats = []
    # Standardize patch size for the model (resize to 224x224 usually best for ViT/ResNet)
    # But for speed, we stick to raw tensor conversion. 
    # Note: Modern CNNs handle variable sizes, but ViTs need fixed size.
    # Here we assume CNN (ResNet/ConvNext) which is robust to size.
    
    for i in range(0, len(patches), batch_size):
        batch_list = patches[i:i+batch_size]
        # Stack and add channel dim: (B, H, W) -> (B, 1, H, W)
        batch = np.stack(batch_list, axis=0)
        x = torch.tensor(batch, dtype=torch.float32, device=device).unsqueeze(1)
        # Repeat to 3 channels for ImageNet models
        x = x.repeat(1, 3, 1, 1)
        
        # Forward pass
        f = model(x).detach().cpu().numpy().astype(np.float32)
        feats.append(f)
        
    return np.concatenate(feats, axis=0)

def normalize(x: np.ndarray) -> np.ndarray:
    """Robust Min-Max normalization."""
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def build_scan_map(img_shape, coords, score):
    """Projects patch scores back onto the image pixels."""
    scan_map = np.zeros(img_shape, dtype=np.float32)
    count = np.zeros(img_shape, dtype=np.float32)
    
    for (i, j, size), s in zip(coords, score):
        scan_map[i:i+size, j:j+size] += float(s)
        count[i:i+size, j:j+size] += 1.0
        
    # Avoid div by zero
    scan_map = scan_map / np.maximum(count, 1.0)
    return scan_map

def compute_top_k(score, coords, k=10):
    """Returns the top K regions based on score."""
    k = int(min(k, len(score)))
    top_idx = np.argsort(score)[-k:][::-1] # Descending
    
    top_regions = []
    for rank, idx in enumerate(top_idx, start=1):
        i, j, size = coords[idx]
        top_regions.append({
            "rank": rank,
            "id": int(idx), # Keep raw index for feedback loop
            "i": int(i),
            "j": int(j),
            "size": int(size),
            "score": float(score[idx]),
        })
    return top_regions

def run_active_learning(
    img: np.ndarray,
    backbone="resnet18",
    patch_sizes=(32, 64),
    strides=(16, 32),
    pca_dim=50,
    umap_neighbors=15,
    umap_min_dist=0.1,
    batch_size=64,
):
    """
    Main pipeline: Extract -> Encode -> Score -> UMAP
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    try:
        model = timm.create_model(backbone, pretrained=True, num_classes=0)
    except:
        # Fallback
        print(f"Backbone {backbone} not found, using resnet18")
        model = timm.create_model("resnet18", pretrained=True, num_classes=0)
        
    model = model.to(device).eval()

    # 1. Extraction & Encoding
    all_features = []
    all_coords = []

    for ps, st in zip(patch_sizes, strides):
        patches, coords = extract_patches(img, ps, st)
        if len(patches) == 0: continue
        
        feats = encode_patches(patches, model, device, batch_size=batch_size)
        all_features.append(feats)
        all_coords.extend(coords)

    if not all_features:
        raise ValueError("No patches extracted. Image might be smaller than patch size.")

    features = np.concatenate(all_features, axis=0).astype(np.float32)

    # 2. Dimensionality Reduction (PCA)
    # We use PCA first to denoise before IsolationForest or UMAP
    pca_dim = min(pca_dim, features.shape[1], features.shape[0])
    if pca_dim > 1:
        pca = PCA(n_components=pca_dim)
        features_reduced = pca.fit_transform(features)
    else:
        features_reduced = features

    # 3. Anomaly Scoring (Isolation Forest)
    # Replaced Mahalanobis with Isolation Forest for robustness
    iso = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    # fit_predict returns -1 for outliers, 1 for inliers. 
    # We want high score for outliers.
    preds = iso.fit_predict(features_reduced)
    # decision_function: lower is more anomalous. We negate it.
    raw_scores = -1 * iso.decision_function(features_reduced)
    
    score = normalize(raw_scores)

    # 4. Generate Maps
    scan_map = build_scan_map(img.shape, all_coords, score)

    # 5. UMAP (for visualization only, done on reduced features)
    reducer = umap.UMAP(n_neighbors=umap_neighbors, min_dist=umap_min_dist, n_components=2, random_state=42)
    embedding = reducer.fit_transform(features_reduced)

    return {
        "device": str(device),
        "features": features_reduced, # Keep reduced features for feedback loop
        "coords": all_coords,
        "score": score,
        "scan_map": scan_map,
        "embedding": embedding,
    }

def rerank_global(features, coords, img_shape, query_idx, mode="similar"):
    """
    Human-in-the-Loop: Re-ranks everything based on similarity to query_idx.
    """
    query_vec = features[query_idx].reshape(1, -1)
    
    # Cosine Similarity: Range [-1, 1]
    sim = cosine_similarity(features, query_vec).flatten()
    
    if mode == "similar":
        # Higher similarity = Higher score
        new_scores = normalize(sim)
    else:
        # Dissimilar (Find anomalies relative to this selection)
        new_scores = normalize(1 - sim)
        
    new_map = build_scan_map(img_shape, coords, new_scores)
    
    return new_scores, new_map
