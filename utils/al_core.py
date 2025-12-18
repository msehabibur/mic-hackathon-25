import numpy as np
import torch
import timm
import umap
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- Helper Utilities ---
def load_image_grayscale(file) -> np.ndarray:
    """Loads image, converts to grayscale, and normalizes to [0,1]."""
    try:
        img = Image.open(file).convert("L")
        img = np.asarray(img, dtype=np.float32)
        denom = (img.max() - img.min())
        if denom < 1e-12: return np.zeros_like(img, dtype=np.float32)
        return (img - img.min()) / denom
    except Exception as e:
        return np.zeros((256, 256), dtype=np.float32)

def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn) if mx > mn else np.zeros_like(x)

def build_scan_map(img_shape, coords, score):
    """Projects patch scores back onto the image pixels."""
    scan_map = np.zeros(img_shape, dtype=np.float32)
    count = np.zeros(img_shape, dtype=np.float32)
    
    for (i, j, size), s in zip(coords, score):
        scan_map[i:i+size, j:j+size] += float(s)
        count[i:i+size, j:j+size] += 1.0
        
    return scan_map / np.maximum(count, 1.0)

def compute_top_k(score, coords, k=10):
    """Returns metadata for the top-k most interesting patches."""
    k = int(min(k, len(score)))
    top_idx = np.argsort(score)[-k:][::-1]
    
    top_regions = []
    for rank, idx in enumerate(top_idx, start=1):
        i, j, size = coords[idx]
        top_regions.append({
            "rank": rank, "id": int(idx),
            "i": int(i), "j": int(j), "size": int(size),
            "score": float(score[idx])
        })
    return top_regions

# --- Core Logic ---

def extract_patches(img, patch_sizes, strides):
    """Extracts patches from image."""
    patches, coords = [], []
    H, W = img.shape
    for ps, st in zip(patch_sizes, strides):
        for i in range(0, H - ps + 1, st):
            for j in range(0, W - ps + 1, st):
                patches.append(img[i:i+ps, j:j+ps])
                coords.append((i, j, ps))
    return patches, coords

@torch.no_grad()
def get_features(patches, backbone, device, batch_size=64):
    """Runs patches through deep learning model."""
    try:
        model = timm.create_model(backbone, pretrained=True, num_classes=0)
    except:
        model = timm.create_model("resnet18", pretrained=True, num_classes=0)
    
    model = model.to(device).eval()
    
    feats = []
    for i in range(0, len(patches), batch_size):
        batch = np.stack(patches[i:i+batch_size], axis=0)
        # Convert grayscale (1ch) to RGB (3ch) for ResNet
        x = torch.tensor(batch, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, 3, 1, 1)
        f = model(x).detach().cpu().numpy()
        feats.append(f)
        
    return np.concatenate(feats, axis=0)

def run_initial_analysis(img, backbone="resnet18", patch_sizes=(32,), strides=(16,), pca_dim=50):
    """Step 1: Unsupervised Anomaly Detection."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Extraction
    patches, coords = extract_patches(img, patch_sizes, strides)
    if not patches: return None
    
    # 2. Featurization
    features = get_features(patches, backbone, device)
    
    # 3. Dimensionality Reduction (PCA)
    pca = PCA(n_components=min(pca_dim, features.shape[0], features.shape[1]))
    features_reduced = pca.fit_transform(features)
    
    # 4. Anomaly Detection (Isolation Forest)
    iso = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    iso.fit(features_reduced)
    raw_scores = -1 * iso.decision_function(features_reduced)
    score = normalize(raw_scores)
    
    # 5. UMAP for visualization
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(features_reduced)
    
    return {
        "features": features_reduced,
        "coords": coords,
        "score": score,
        "scan_map": build_scan_map(img.shape, coords, score),
        "embedding": embedding,
        "device": str(device)
    }

def train_active_learner(features, coords, img_shape, positive_idx, negative_indices=None):
    """
    Step 2: Supervised Active Learning (Logistic Regression).
    Trains a classifier to distinguish 'Positive' patch from 'Negative' patches.
    """
    N = features.shape[0]
    
    # 1. Build Dataset
    X_train = [features[positive_idx]]
    y_train = [1] # 1 = Interesting

    # If no negatives provided, sample random background patches
    if negative_indices is None or len(negative_indices) == 0:
        import random
        # Sample 5 random patches, ensuring we don't pick the positive one
        candidates = list(set(range(N)) - {positive_idx})
        neg_idxs = random.sample(candidates, min(5, len(candidates)))
    else:
        neg_idxs = negative_indices

    for n_idx in neg_idxs:
        X_train.append(features[n_idx])
        y_train.append(0) # 0 = Boring

    # 2. Train Classifier
    # Class_weight='balanced' handles the imbalance between few positives and many negatives
    clf = LogisticRegression(class_weight='balanced', C=10.0, solver='lbfgs')
    clf.fit(X_train, y_train)

    # 3. Predict for WHOLE image
    probs = clf.predict_proba(features)[:, 1] # Probability of being 'Interesting'
    
    return normalize(probs), build_scan_map(img_shape, coords, probs)
