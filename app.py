import io
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import torch
import timm
import umap
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. CORE UTILITIES & AI LOGIC
# -----------------------------------------------------------------------------

def load_image_grayscale(file) -> np.ndarray:
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
    if mx - mn < 1e-8: return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def build_scan_map(img_shape, coords, score):
    scan_map = np.zeros(img_shape, dtype=np.float32)
    count = np.zeros(img_shape, dtype=np.float32)
    for (i, j, size), s in zip(coords, score):
        scan_map[i:i+size, j:j+size] += float(s)
        count[i:i+size, j:j+size] += 1.0
    return scan_map / np.maximum(count, 1.0)

def extract_patches_single_size(img, patch_size, stride):
    """
    Extracts patches of a SINGLE fixed size.
    """
    patches, coords = [], []
    H, W = img.shape
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patches.append(img[i:i+patch_size, j:j+patch_size])
            coords.append((i, j, patch_size))
    return patches, coords

@torch.no_grad()
def get_features(patches, backbone_name, device, batch_size=32):
    """
    Extracts features. Resizes all patches to 224x224 to keep the model happy.
    """
    if not patches: return np.empty((0, 0))

    # Load Model
    if "dino" in backbone_name:
        try:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        except:
            st.warning("⚠️ DINOv2 failed to load, using ResNet18.")
            model = timm.create_model("resnet18", pretrained=True, num_classes=0)
    else:
        try:
            model = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        except:
            model = timm.create_model("resnet18", pretrained=True, num_classes=0)

    model = model.to(device).eval()
    
    feats = []
    # Process in batches
    for i in range(0, len(patches), batch_size):
        batch_list = patches[i:i+batch_size]
        
        # 1. Stack numpy arrays (They are guaranteed to be same size now)
        batch_np = np.stack(batch_list, axis=0)
        
        # 2. Convert to Torch (B, 1, H, W) -> Repeat to RGB (B, 3, H, W)
        x = torch.tensor(batch_np, dtype=torch.float32, device=device).unsqueeze(1)
        x = x.repeat(1, 3, 1, 1) 
        
        # 3. Resize to standard 224x224 (Critical for model stability)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        output = model(x)
        
        # Handle DINO vs ResNet output formats
        if isinstance(output, dict):
            f = output['x_norm_clstoken'].detach().cpu().numpy()
        else:
            f = output.detach().cpu().numpy()
            
        feats.append(f)
        
    return np.concatenate(feats, axis=0) if feats else np.empty((0,0))

def run_analysis_pipeline(img, backbone, patch_sizes, strides, pca_dim=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_features = []
    all_coords = []
    
    # LOOP over sizes separately to avoid shape mismatch errors
    for ps, st in zip(patch_sizes, strides):
        # 1. Extract
        p_curr, c_curr = extract_patches_single_size(img, ps, st)
        if not p_curr: continue
        
        # 2. Get Features (returns (N, D) array)
        f_curr = get_features(p_curr, backbone, device)
        
        all_features.append(f_curr)
        all_coords.extend(c_curr)
    
    if not all_features:
        return None

    # Concatenate features from different patch sizes
    features = np.concatenate(all_features, axis=0)
    
    # 3. PCA
    n_samples, n_dim = features.shape
    pca_n = min(pca_dim, n_samples, n_dim)
    if pca_n > 1:
        pca = PCA(n_components=pca_n)
        features_reduced = pca.fit_transform(features)
    else:
        features_reduced = features
    
    # 4. Anomaly Detection (Isolation Forest)
    iso = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    iso.fit(features_reduced)
    score = normalize(-1 * iso.decision_function(features_reduced))
    
    # 5. Embedding (UMAP) - Fallback if data is too small
    try:
        embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2).fit_transform(features_reduced)
    except:
        embedding = np.zeros((len(features), 2))
        if features_reduced.shape[1] >= 2:
            embedding = features_reduced[:, :2]

    return {
        "features": features_reduced,
        "coords": all_coords,
        "score": score,
        "scan_map": build_scan_map(img.shape, all_coords, score),
        "embedding": embedding,
        "device": str(device)
    }

def train_classifier(features, coords, img_shape, pos_idx, neg_indices=None):
    """Refines the heatmap using Logistic Regression (Active Learning)."""
    # Safety check
    if pos_idx >= len(features): return np.zeros(len(features)), np.zeros(img_shape)

    X_train = [features[pos_idx]]
    y_train = [1]
    
    # Negative Mining
    if not neg_indices:
        import random
        # Pick random indices that are NOT the positive one
        candidates = list(range(len(features)))
        candidates.remove(pos_idx)
        neg_indices = random.sample(candidates, min(5, len(candidates)))
    
    for idx in neg_indices:
        X_train.append(features[idx])
        y_train.append(0)
        
    clf = LogisticRegression(class_weight='balanced', C=10.0, solver='lbfgs')
    clf.fit(X_train, y_train)
    
    probs = clf.predict_proba(features)[:, 1]
    return normalize(probs), build_scan_map(img_shape, coords, probs)

# -----------------------------------------------------------------------------
# 2. STREAMLIT UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Microscopy Active Learning", layout="wide", page_icon="🔬")
st.title("🔬 Next-Best-Scan: Interactive Discovery")

if "results" not in st.session_state: st.session_state.results = None
if "img_cache" not in st.session_state: st.session_state.img_cache = None
if "mode" not in st.session_state: st.session_state.mode = "Unsupervised"
if "history" not in st.session_state: st.session_state.history = []

# --- SIDEBAR ---
st.sidebar.header("1. Upload & Settings")
uploaded = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "tif"])
backbone = st.sidebar.selectbox("Model Backbone", ["resnet18", "resnet50", "dinov2_vits14 (Meta AI)"], index=0)

run_btn = st.sidebar.button("🚀 Run Analysis")

# --- MAIN EXECUTION ---
if uploaded:
    img_bytes = uploaded.getvalue()
    img = load_image_grayscale(io.BytesIO(img_bytes))
    
    # Reset if new image
    if st.session_state.img_cache is None or not np.array_equal(img, st.session_state.img_cache):
        st.session_state.img_cache = img
        st.session_state.results = None
        st.session_state.history = []

    if run_btn:
        with st.spinner(f"Running analysis with {backbone}..."):
            # Map selection to actual model name
            model_name = "dinov2_vits14" if "dino" in backbone else backbone
            
            # Pass patch sizes as tuple
            res = run_analysis_pipeline(img, model_name, (32, 64), (16, 32))
            
            if res:
                st.session_state.results = res
                st.session_state.current_score = res["score"]
                st.session_state.current_map = res["scan_map"]
                st.session_state.mode = "Unsupervised"
                st.session_state.history.append(res["scan_map"]) # Save baseline
            else:
                st.error("Could not extract patches. Image might be too small.")

if st.session_state.results is not None:
    res = st.session_state.results
    img = st.session_state.img_cache
    score = st.session_state.current_score
    
    # Helper to get Top K
    k = 10
    top_idx = np.argsort(score)[-k:][::-1]
    top_regions = []
    for r, idx in enumerate(top_idx):
        i, j, s = res["coords"][idx]
        top_regions.append({"rank": r+1, "id": idx, "i": i, "j": j, "size": s, "score": score[idx]})

    # --- ROW 1: VISUALIZATION ---
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Input Image")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, cmap="gray")
        for r in top_regions:
            rect = mpatches.Rectangle((r["j"], r["i"]), r["size"], r["size"], linewidth=2, edgecolor="lime", facecolor="none")
            ax.add_patch(rect)
            ax.text(r["j"], max(0, r["i"]-5), str(r["rank"]), color="lime", weight="bold")
        ax.axis("off")
        st.pyplot(fig)

    with c2:
        st.subheader(f"Priority Map ({st.session_state.mode})")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, cmap="gray", alpha=0.4)
        im = ax.imshow(st.session_state.current_map, cmap="jet", alpha=0.6)
        ax.axis("off")
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)

    st.divider()

    # --- ROW 2: TEACHER MODE ---
    col_teach, col_plot = st.columns([1, 2])
    
    with col_teach:
        st.header("🧠 Teacher Mode")
        sel_rank = st.selectbox("Select Region of Interest:", [r["rank"] for r in top_regions])
        target = next(r for r in top_regions if r["rank"] == sel_rank)
        
        if st.button("Find More Like This 🔍"):
            new_score, new_map = train_classifier(res["features"], res["coords"], img.shape, target["id"])
            st.session_state.current_score = new_score
            st.session_state.current_map = new_map
            st.session_state.mode = f"Targeting Region #{sel_rank}"
            st.session_state.history.append(new_map) # Save for efficiency tracking
            st.rerun()

        if st.button("Reset 🔄"):
            st.session_state.current_score = res["score"]
            st.session_state.current_map = res["scan_map"]
            st.session_state.mode = "Unsupervised"
            # Keep only baseline in history or reset? Let's reset history to baseline
            if st.session_state.history:
                 st.session_state.history = [st.session_state.history[0]]
            st.rerun()

    with col_plot:
        st.subheader("Patch Similarity Space")
        # Safety for UMAP
        embed_data = res["embedding"]
        if embed_data.shape[0] != len(score):
            st.warning("Embedding size mismatch.")
        else:
            df = pd.DataFrame(embed_data, columns=["x", "y"])
            df["score"] = score
            df["size"] = 2
            if target["id"] < len(df):
                df.loc[target["id"], "size"] = 10
                
            fig = px.scatter(df, x="x", y="y", color="score", size="size", color_continuous_scale="Jet")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- ROW 3: EFFICIENCY METRICS (The Award Winner) ---
    st.header("📊 Simulated Efficiency Gains")
    
    # Logic for metrics
    def calculate_metrics(scan_map, threshold=10):
        flat = scan_map.flatten()
        n = len(flat)
        if n == 0: return 0
        sorted_px = np.sort(flat)[::-1]
        total_sig = sorted_px.sum()
        if total_sig == 0: return 0
        
        # Signal captured at X% pixels
        cutoff = int(n * (threshold / 100))
        captured = sorted_px[:cutoff].sum()
        return (captured / total_sig) * 100

    latest_map = st.session_state.current_map
    
    m1, m2, m3 = st.columns(3)
    
    # 1. Signal at 10% Scan
    sig_10 = calculate_metrics(latest_map, 10)
    m1.metric("Signal Captured (10% Scan)", f"{sig_10:.1f}%", delta=f"{sig_10 - 10:.1f}% vs Random")
    
    # 2. Time to 80% Signal
    # Find how many pixels needed to get 80% signal
    flat = np.sort(latest_map.flatten())[::-1]
    cum = np.cumsum(flat)
    total = cum[-1] if len(cum) > 0 else 1
    idx_80 = np.searchsorted(cum, 0.8 * total)
    time_pct = (idx_80 / len(flat)) * 100 if len(flat) > 0 else 100
    
    m2.metric("Time to 80% Quality", f"{time_pct:.1f}%", delta=f"-{100-time_pct:.1f}% Time Saved")
    
    # 3. Improvement
    m3.metric("Training Steps", len(st.session_state.history))

    # Efficiency Curve
    x_vals = np.linspace(0, 100, 50)
    y_vals = [calculate_metrics(latest_map, x) for x in x_vals]
    
    fig_eff, ax_eff = plt.subplots(figsize=(10, 3))
    ax_eff.plot(x_vals, y_vals, color="red", label="AI Adaptive Scan")
    ax_eff.plot(x_vals, x_vals, color="gray", linestyle="--", label="Random Scan")
    ax_eff.fill_between(x_vals, y_vals, x_vals, color="red", alpha=0.1)
    ax_eff.set_ylabel("% Information Found")
    ax_eff.set_xlabel("% Time Spent")
    ax_eff.legend()
    ax_eff.grid(True, alpha=0.3)
    st.pyplot(fig_eff)

elif not uploaded:
    st.info("👈 Please upload an image to start.")
