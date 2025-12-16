import numpy as np
import torch
import timm
import umap
import cv2
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import SAM

# Ensure consistent float type
torch.set_default_dtype(torch.float32)

# --- CORE UTILS ---
def load_image_grayscale(file) -> np.ndarray:
    """Loads image, converts to grayscale, and normalizes to [0,1]."""
    try:
        img_pil = Image.open(file).convert("L")
        img = np.asarray(img_pil, dtype=np.float32)
        denom = (img.max() - img.min())
        if denom < 1e-12:
            return np.zeros_like(img, dtype=np.float32), img_pil
        img = (img - img.min()) / denom
        return img, img_pil
    except Exception as e:
        print(f"Error loading image: {e}")
        return np.zeros((256, 256), dtype=np.float32), Image.new("L", (256, 256))

def extract_patches(img: np.ndarray, patch_size: int, stride: int):
    patches, coords = [], []
    H, W = img.shape
    if H < patch_size or W < patch_size: return [], []
        
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patches.append(img[i:i+patch_size, j:j+patch_size])
            # i=y, j=x, size
            coords.append((i, j, patch_size))
    return patches, coords

@torch.no_grad()
def encode_patches(patches, model, device, batch_size=64):
    if not patches: return np.empty((0, 0))
    feats = []
    for i in range(0, len(patches), batch_size):
        batch_list = patches[i:i+batch_size]
        batch = np.stack(batch_list, axis=0)
        x = torch.tensor(batch, dtype=torch.float32, device=device).unsqueeze(1)
        x = x.repeat(1, 3, 1, 1) # 3 channels
        f = model(x).detach().cpu().numpy().astype(np.float32)
        feats.append(f)
    return np.concatenate(feats, axis=0)

def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8: return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def build_scan_map(img_shape, coords, score):
    scan_map = np.zeros(img_shape, dtype=np.float32)
    count = np.zeros(img_shape, dtype=np.float32)
    for (i, j, size), s in zip(coords, score):
        scan_map[i:i+size, j:j+size] += float(s)
        count[i:i+size, j:j+size] += 1.0
    return scan_map / np.maximum(count, 1.0)

def compute_top_k(score, coords, k=10):
    k = int(min(k, len(score)))
    top_idx = np.argsort(score)[-k:][::-1]
    top_regions = []
    for rank, idx in enumerate(top_idx, start=1):
        i, j, size = coords[idx]
        top_regions.append({
            "rank": rank, "id": int(idx), "i": int(i), "j": int(j), "size": int(size), "score": float(score[idx]),
        })
    return top_regions

# --- MAIN PIPELINE ---
def run_active_learning(img: np.ndarray, backbone="resnet18", patch_sizes=(32, 64), strides=(16, 32), pca_dim=50, umap_neighbors=15, umap_min_dist=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = timm.create_model(backbone, pretrained=True, num_classes=0)
    except:
        model = timm.create_model("resnet18", pretrained=True, num_classes=0)
    model = model.to(device).eval()

    all_features, all_coords = [], []
    for ps, st in zip(patch_sizes, strides):
        patches, coords = extract_patches(img, ps, st)
        if not patches: continue
        feats = encode_patches(patches, model, device)
        all_features.append(feats)
        all_coords.extend(coords)

    if not all_features: raise ValueError("No patches extracted.")
    features = np.concatenate(all_features, axis=0).astype(np.float32)

    # PCA & Isolation Forest
    pca_dim = min(pca_dim, features.shape[1], features.shape[0])
    features_reduced = PCA(n_components=pca_dim).fit_transform(features) if pca_dim > 1 else features
    
    iso = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    iso.fit(features_reduced)
    score = normalize(-1 * iso.decision_function(features_reduced))

    scan_map = build_scan_map(img.shape, all_coords, score)
    embedding = umap.UMAP(n_neighbors=umap_neighbors, min_dist=umap_min_dist, n_components=2, random_state=42).fit_transform(features_reduced)

    return {
        "device": str(device), "features": features_reduced, "coords": all_coords,
        "score": score, "scan_map": scan_map, "embedding": embedding,
    }

# --- INTERACTIVE FEATURES ---

def rerank_global(features, coords, img_shape, query_idx, mode="similar"):
    """Simple cosine similarity re-ranking."""
    query_vec = features[query_idx].reshape(1, -1)
    sim = cosine_similarity(features, query_vec).flatten()
    new_scores = normalize(sim) if mode == "similar" else normalize(1 - sim)
    return new_scores, build_scan_map(img_shape, coords, new_scores)

def train_interactive_model(features, coords, img_shape, pos_idx, neg_idx):
    """Trains a Logistic Regression classifier on user-labeled patch indices."""
    if not pos_idx or not neg_idx:
        return None, None
        
    X_pos = features[pos_idx]
    y_pos = np.ones(len(pos_idx))
    X_neg = features[neg_idx]
    y_neg = np.zeros(len(neg_idx))
    
    X_train = np.concatenate([X_pos, X_neg])
    y_train = np.concatenate([y_pos, y_neg])
    
    # Balanced class weights handle unequal number of clicks
    clf = LogisticRegression(class_weight='balanced', solver='liblinear', C=1.0)
    clf.fit(X_train, y_train)
    
    # Predict probability of being "positive" class
    probs = clf.predict_proba(features)[:, 1]
    normalized_probs = normalize(probs)
    
    classifier_map = build_scan_map(img_shape, coords, normalized_probs)
    return normalized_probs, classifier_map

def generate_sam_masks(img_pil_orig, top_regions):
    """
    Uses MobileSAM to generate segmentation masks from bounding box prompts.
    """
    try:
        # Downloads lightweight model automatically
        model = SAM('mobile_sam.pt') 
    except Exception as e:
        print(f"SAM model failed to load: {e}")
        return []

    img_rgb = img_pil_orig.convert("RGB")
    img_np = np.array(img_rgb)
    
    masks = []
    for r in top_regions:
        bbox = [r["j"], r["i"], r["j"] + r["size"], r["i"] + r["size"]]
        results = model.predict(img_np, bboxes=[bbox], verbose=False)
        
        if results[0].masks is not None:
            mask_data = results[0].masks.data[0].cpu().numpy().astype(np.uint8)
            # Resize mask if needed
            if mask_data.shape != img_np.shape[:2]:
                 mask_data = cv2.resize(mask_data, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            masks.append(mask_data)
        else:
            masks.append(None)
    return masks
