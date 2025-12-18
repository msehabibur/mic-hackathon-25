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
from skimage import data, util # For default demo image

# New Import for Text Search
from transformers import CLIPProcessor, CLIPModel

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DeepScan Pro: Intelligent Microscopy",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. CORE UTILITIES & AI LOGIC
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

def get_default_image() -> np.ndarray:
    """Loads a default STEM-like example if no file is uploaded."""
    # We use a scientific sample from scikit-image (e.g., micro-structure)
    img = util.img_as_float(data.brick()) # 'brick' has good texture for anomaly detection
    return img

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
    patches, coords = [], []
    H, W = img.shape
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patches.append(img[i:i+patch_size, j:j+patch_size])
            coords.append((i, j, patch_size))
    return patches, coords

# --- CACHED MODEL LOADING ---
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

@torch.no_grad()
def get_features(patches, backbone_name, device, batch_size=32):
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

@torch.no_grad()
def search_by_text(patches, text_query, device="cpu", batch_size=32):
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

    return np.concatenate(scores)

def run_analysis_pipeline(img, backbone, patch_sizes, strides, pca_dim=50):
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
    
    n_samples, n_dim = features.shape
    pca_n = min(pca_dim, n_samples, n_dim)
    features_reduced = PCA(n_components=pca_n).fit_transform(features) if pca_n > 1 else features
    
    iso = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    iso.fit(features_reduced)
    score = normalize(-1 * iso.decision_function(features_reduced))
    
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
    return normalize(probs), build_scan_map(img_shape, coords, probs)

# -----------------------------------------------------------------------------
# 3. STREAMLIT UI STRUCTURE
# -----------------------------------------------------------------------------

# Sidebar Navigation
st.sidebar.title("🧬 Navigator")
page_selection = st.sidebar.radio("Go to:", ["🚀 Main Application", "📘 The Math Behind It"])

# --- PAGE 1: MAIN APPLICATION ---
if page_selection == "🚀 Main Application":
    st.title("🔬 DeepScan Pro: Intelligent Microscopy")

    # Initialize State
    if "results" not in st.session_state: st.session_state.results = None
    if "img_cache" not in st.session_state: st.session_state.img_cache = None
    if "history" not in st.session_state: st.session_state.history = []

    # Sidebar Inputs
    st.sidebar.header("1. Input & Settings")
    uploaded = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "tif"])
    
    # --- LOGIC: Load Uploaded OR Default ---
    if uploaded:
        img_bytes = uploaded.getvalue()
        img = load_image_grayscale(io.BytesIO(img_bytes))
        st.sidebar.success("✅ Custom Image Loaded")
    elif st.session_state.img_cache is not None:
        img = st.session_state.img_cache # Keep using what we have
    else:
        # Load Default Demo Image on Startup
        img = get_default_image()
        st.session_state.img_cache = img
        st.sidebar.info("ℹ️ Using Default Demo Image")

    # Update cache if image changed
    if st.session_state.img_cache is None or not np.array_equal(img, st.session_state.img_cache):
         st.session_state.img_cache = img
         st.session_state.results = None
         st.session_state.history = []

    # Settings
    backbone = st.sidebar.selectbox("Vision Backbone", ["regnet_y_400mf", "convnext_tiny", "resnet50"], index=1)
    
    with st.sidebar.expander("🛠️ Advanced Config"):
        patch_size_input = st.multiselect("Patch Sizes", [32, 64, 128], default=[32, 64])
        stride_val = st.slider("Overlap (Stride divisor)", 1, 4, 2)
        pca_dim = st.slider("PCA Dim", 10, 100, 50)

    run_btn = st.sidebar.button("🚀 Run Analysis")
    
    if st.sidebar.button("Reset 🗑️"):
        st.session_state.clear()
        st.rerun()

    # Execution
    if run_btn:
        with st.spinner(f"Analyzing with {backbone}..."):
            strides = tuple([p // stride_val for p in patch_size_input])
            res = run_analysis_pipeline(img, backbone, tuple(patch_size_input), strides, pca_dim=pca_dim)
            if res:
                st.session_state.results = res
                st.session_state.current_score = res["score"]
                st.session_state.current_map = res["scan_map"]
                st.session_state.mode = "Unsupervised"
                st.session_state.history.append(res["scan_map"])
            else:
                st.error("Image too small.")

    # Results Display
    if st.session_state.results is not None:
        res = st.session_state.results
        score = st.session_state.current_score
        
        # Top 10 Regions
        top_idx = np.argsort(score)[-10:][::-1]
        top_regions = [{"rank": r+1, "id": i, "i": res["coords"][i][0], "j": res["coords"][i][1], "size": res["coords"][i][2]} for r, i in enumerate(top_idx)]

        tab1, tab2, tab3 = st.tabs(["👁️ Visual Scan", "💬 Text Search", "📊 Efficiency Report"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Microscope View & Path")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(img, cmap="gray")
                # Draw Path
                path_y = [r["i"] + r["size"]//2 for r in top_regions]
                path_x = [r["j"] + r["size"]//2 for r in top_regions]
                ax.plot(path_x, path_y, 'r--', linewidth=1.5, alpha=0.8, label="Scan Path")
                ax.scatter(path_x[0], path_y[0], c='lime', s=100, zorder=5, label="Start")
                for r in top_regions:
                    rect = mpatches.Rectangle((r["j"], r["i"]), r["size"], r["size"], linewidth=2, edgecolor="lime", facecolor="none")
                    ax.add_patch(rect)
                ax.legend(loc="lower right")
                ax.axis("off")
                st.pyplot(fig)

            with c2:
                st.subheader(f"Heatmap ({st.session_state.mode})")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(img, cmap="gray", alpha=0.4)
                im = ax.imshow(st.session_state.current_map, cmap="jet", alpha=0.6)
                ax.axis("off")
                plt.colorbar(im, ax=ax)
                st.pyplot(fig)
            
            st.divider()
            
            # Interactive Feedback
            col_teach, col_plot = st.columns([1, 2])
            with col_teach:
                st.markdown("### 👨‍🏫 Teach the AI")
                sel_rank = st.selectbox("Interesting Region:", [r["rank"] for r in top_regions])
                target = next(r for r in top_regions if r["rank"] == sel_rank)
                
                if st.button("Find More Like This 🔍"):
                    new_score, new_map = train_classifier(res["features"], res["coords"], img.shape, target["id"])
                    st.session_state.current_score = new_score
                    st.session_state.current_map = new_map
                    st.session_state.mode = f"Supervised (Like #{sel_rank})"
                    st.session_state.history.append(new_map)
                    st.rerun()
            
            with col_plot:
                # UMAP
                embed_data = res["embedding"]
                if embed_data.shape[0] == len(score):
                    df = pd.DataFrame(embed_data, columns=["x", "y"])
                    df["score"] = score
                    df["size"] = 2
                    if target["id"] < len(df): df.loc[target["id"], "size"] = 10
                    fig = px.scatter(df, x="x", y="y", color="score", size="size", color_continuous_scale="Jet", title="Latent Feature Space")
                    fig.update_layout(height=300, margin=dict(t=30, l=0, r=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("### 🗣️ Semantic Search")
            st.info("Type a query to search the image using Multimodal CLIP AI.")
            query = st.text_input("Query:", placeholder="e.g. 'dark circular defects' or 'linear cracks'")
            if st.button("Search 🔎"):
                with st.spinner("CLIP is seeing the image..."):
                    text_scores = search_by_text(res["raw_patches"], query)
                    text_scores = normalize(text_scores)
                    text_map = build_scan_map(img.shape, res["coords"], text_scores)
                    st.session_state.current_score = text_scores
                    st.session_state.current_map = text_map
                    st.session_state.mode = f"Text: '{query}'"
                    st.session_state.history.append(text_map)
                    st.rerun()

        with tab3:
            st.markdown("### 📊 Efficiency Report")
            
            def calc_metrics(scan_map, t=10):
                flat = np.sort(scan_map.flatten())[::-1]
                total = flat.sum() + 1e-9
                cutoff = int(len(flat) * (t/100))
                return (flat[:cutoff].sum() / total) * 100

            curr = st.session_state.current_map
            eff_10 = calc_metrics(curr, 10)
            
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Signal @ 10% Scan", f"{eff_10:.1f}%", f"+{eff_10-10:.1f}%")
            col_m2.metric("Steps Taken", len(st.session_state.history))
            
            x_vals = np.linspace(0, 100, 50)
            y_vals = [calc_metrics(curr, x) for x in x_vals]
            
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(x_vals, y_vals, 'r-', linewidth=2, label="AI Scan")
            ax.plot(x_vals, x_vals, 'k--', alpha=0.3, label="Random")
            ax.fill_between(x_vals, y_vals, x_vals, color='red', alpha=0.1)
            ax.set_ylabel("% Anomalies Found")
            ax.set_xlabel("% Time Spent")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Download
            df_rep = pd.DataFrame({"Scan_Percentage": x_vals, "Signal_Captured": y_vals})
            st.download_button("Download CSV 📄", df_rep.to_csv(index=False), "efficiency_report.csv", "text/csv")
            
    elif not uploaded and st.session_state.img_cache is None:
         st.warning("⚠️ Loading demo image...")
         st.rerun()

# --- PAGE 2: THE MATH BEHIND IT ---
elif page_selection == "📘 The Math Behind It":
    st.title("📘 The Mathematics of DeepScan")
    st.markdown("This application uses a pipeline of **Self-Supervised Learning** and **Active Learning**. Here is the mathematical foundation.")
    
    st.header("1. Feature Extraction (RegNet/ConvNeXt)")
    st.markdown(r"""
    

[Image of neural network convolution diagram]

    
    We map an image patch $x \in \mathbb{R}^{H \times W}$ to a feature vector $z \in \mathbb{R}^d$ using a Convolutional Neural Network (CNN) $f_\theta$:
    $$ z = f_\theta(x) $$
    These features capture texture and geometry invariant to rotation or slight shifts.
    """)
    
    st.divider()
    
    st.header("2. Anomaly Detection (Isolation Forest)")
    st.markdown(r"""
    
    
    To find interesting regions without labels, we use **Isolation Forests**. The core idea is that *anomalies are easier to isolate* (require fewer random cuts).
    The anomaly score $s(x, n)$ for a sample $x$ in a dataset of size $n$ is:
    
    $$ s(x, n) = 2^{- \frac{E(h(x))}{c(n)}} $$
    
    Where:
    * $h(x)$ is the path length (number of splits) to isolate sample $x$.
    * $E(h(x))$ is the average path length across a forest of random trees.
    * $c(n)$ is a normalization factor (average path length of a binary search tree).
    
    If $s(x, n) \to 1$, the sample is an **anomaly** (interesting).
    If $s(x, n) \to 0.5$, the sample is **background** (boring).
    """)

    st.divider()

    st.header("3. Semantic Search (CLIP)")
    st.markdown(r"""
    
    
    For text search, we use **CLIP (Contrastive Language-Image Pre-training)**. It aligns image embeddings $I_f$ and text embeddings $T_f$ in a shared space by maximizing the cosine similarity for correct pairs:
    
    $$ \text{similarity}(I, T) = \frac{I_f \cdot T_f}{\|I_f\| \|T_f\|} $$
    
    The loss function minimizes the distance between the user's query ("defects") and the matching image patches.
    """)
