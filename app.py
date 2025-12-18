import io
import os
import json
import time
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
from skimage import data, util, measure 
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

def load_image_grayscale(file_or_path) -> np.ndarray:
    try:
        if isinstance(file_or_path, str):
            img = Image.open(file_or_path).convert("L")
        else:
            img = Image.open(file_or_path).convert("L")
            
        img = np.asarray(img, dtype=np.float32)
        denom = (img.max() - img.min())
        if denom < 1e-12: return np.zeros_like(img, dtype=np.float32)
        return (img - img.min()) / denom
    except Exception as e:
        return np.zeros((256, 256), dtype=np.float32)

def get_default_image() -> np.ndarray:
    possible_names = ["STEM_example.png", "STEM_example.jpg", "STEM_example.tif"]
    for fname in possible_names:
        if os.path.exists(fname):
            return load_image_grayscale(fname)
    return util.img_as_float(data.brick())

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

st.sidebar.title("🧬 Navigator")
page_selection = st.sidebar.radio("Go to:", ["🚀 Main Application", "📘 The Math Behind It"])

# --- PAGE 1: MAIN APPLICATION ---
if page_selection == "🚀 Main Application":
    st.title("🔬 DeepScan Pro: Intelligent Microscopy")
    
    if "results" not in st.session_state: st.session_state.results = None
    if "img_cache" not in st.session_state: st.session_state.img_cache = None
    if "history" not in st.session_state: st.session_state.history = []
    if "connected" not in st.session_state: st.session_state.connected = False

    # --- HARDWARE CONNECTION MOCK ---
    st.sidebar.divider()
    st.sidebar.subheader("Hardware Status")
    if not st.session_state.connected:
        st.sidebar.error("🔴 Disconnected")
        if st.sidebar.button("🔌 Connect Microscope"):
            with st.spinner("Handshaking with Controller..."):
                time.sleep(1.0)
            st.session_state.connected = True
            st.rerun()
    else:
        st.sidebar.success("🟢 Connected (STEM-v3)")
        if st.sidebar.button("❌ Disconnect"):
            st.session_state.connected = False
            st.rerun()
    st.sidebar.divider()

    st.sidebar.header("1. Input & Settings")
    uploaded = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "tif"])
    
    if uploaded:
        img_bytes = uploaded.getvalue()
        img = load_image_grayscale(io.BytesIO(img_bytes))
        st.sidebar.success("✅ Custom Image Loaded")
    elif st.session_state.img_cache is not None:
        img = st.session_state.img_cache 
    else:
        img = get_default_image()
        st.session_state.img_cache = img
        st.sidebar.info("ℹ️ Loaded Default STEM_example")

    if st.session_state.img_cache is None or not np.array_equal(img, st.session_state.img_cache):
         st.session_state.img_cache = img
         st.session_state.results = None
         st.session_state.history = []

    backbone = st.sidebar.selectbox("Vision Backbone", ["regnet_y_400mf", "convnext_tiny", "resnet50"], index=1)
    
    with st.sidebar.expander("📏 Calibration (Physical Units)"):
        nm_per_pixel = st.number_input("Scale (nm per pixel)", value=1.0, min_value=0.01, format="%.2f")
        st.caption(f"100 pixels = {100*nm_per_pixel:.1f} nm")
        
    with st.sidebar.expander("🛠️ Advanced Config"):
        patch_size_input = st.multiselect("Patch Sizes", [32, 64, 128], default=[32, 64])
        stride_val = st.slider("Overlap (Stride divisor)", 1, 4, 2)
        pca_dim = st.slider("PCA Dim", 10, 100, 50)

    run_btn = st.sidebar.button("🚀 Run Analysis")
    
    if st.sidebar.button("Reset 🗑️"):
        st.session_state.clear()
        st.rerun()

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

    if st.session_state.results is not None:
        res = st.session_state.results
        score = st.session_state.current_score
        
        top_idx = np.argsort(score)[-10:][::-1]
        top_regions = [{"rank": r+1, "id": i, "i": res["coords"][i][0], "j": res["coords"][i][1], "size": res["coords"][i][2]} for r, i in enumerate(top_idx)]

        tab1, tab2, tab3, tab4 = st.tabs(["👁️ Visual Scan", "📏 Quantification", "💬 Text Search", "📊 Efficiency"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Microscope View & Path")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(img, cmap="gray")
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
                
                # PROTOCOL DOWNLOAD with BALLOONS
                protocol_data = {
                    "timestamp": "2023-10-27T10:00:00",
                    "scale_nm": nm_per_pixel,
                    "scan_points": [{"x": int(r['j']), "y": int(r['i']), "priority": int(r['rank'])} for r in top_regions]
                }
                if st.download_button("💾 Download Scan Protocol (.json)", json.dumps(protocol_data, indent=2), "scan_protocol.json", "application/json"):
                    st.balloons()
                    st.success("Protocol exported successfully!")


            with c2:
                st.subheader(f"Heatmap ({st.session_state.mode})")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(img, cmap="gray", alpha=0.4)
                im = ax.imshow(st.session_state.current_map, cmap="jet", alpha=0.6)
                ax.axis("off")
                plt.colorbar(im, ax=ax)
                st.pyplot(fig)
            
            # --- DEFECT GALLERY ---
            st.divider()
            st.subheader("🔍 Identified Anomalies (Zoomed Views)")
            cols = st.columns(5)
            for idx, col in enumerate(cols):
                if idx < 5 and idx < len(top_regions):
                    r = top_regions[idx]
                    patch_img = img[r['i']:r['i']+r['size'], r['j']:r['j']+r['size']]
                    col.image(patch_img, caption=f"Rank #{r['rank']}", use_column_width=True, clamp=True)
            st.divider()

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
            st.markdown("### 📏 Defect Quantification")
            st.info("Set a threshold to automatically count and measure defects.")
            
            c_thresh, c_metrics = st.columns([1, 1])
            with c_thresh:
                threshold = st.slider("Anomaly Confidence Threshold", 0.0, 1.0, 0.6, 0.05)
                binary_map = st.session_state.current_map > threshold
                labels = measure.label(binary_map)
                n_features = labels.max()
                
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(img, cmap="gray", alpha=0.6)
                ax.imshow(binary_map, cmap="Reds", alpha=0.5)
                ax.axis("off")
                ax.set_title(f"Detected Regions (> {threshold})")
                st.pyplot(fig)
                
            with c_metrics:
                st.metric("Detected Features", int(n_features))
                total_pixels = binary_map.sum()
                area_nm2 = total_pixels * (nm_per_pixel ** 2)
                if area_nm2 > 1000:
                    st.metric("Total Defect Area", f"{area_nm2/1000:.2f} µm²")
                else:
                    st.metric("Total Defect Area", f"{area_nm2:.1f} nm²")
                st.metric("Coverage", f"{(total_pixels / binary_map.size)*100:.1f}% of surface")

        with tab3:
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

        with tab4:
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
            
            df_rep = pd.DataFrame({"Scan_Percentage": x_vals, "Signal_Captured": y_vals})
            if st.download_button("Download CSV 📄", df_rep.to_csv(index=False), "efficiency_report.csv", "text/csv"):
                st.balloons()

    elif not uploaded and st.session_state.img_cache is None:
         st.warning("⚠️ Loading demo image...")
         st.rerun()

# --- PAGE 2: THE MATH BEHIND IT (FIXED FOR CLEAN DISPLAY) ---
elif page_selection == "📘 The Math Behind It":
    st.title("📘 The Science of DeepScan")
    st.markdown("""
    This platform transforms a microscope into an intelligent agent using a pipeline of **Computer Vision**, 
    **Dimensionality Reduction**, and **Active Learning**. Here is the deep technical breakdown.
    """)
    
    st.divider()

    st.header("1. Sliding Window & Feature Extraction")
    st.markdown("""
    The high-resolution microscope image $I$ is essentially a massive matrix of pixels. 
    We cannot process it all at once, so we decompose it into small overlapping patches.
    
    **The Neural Backbone:**
    We use **RegNet** or **ConvNeXt**, which are modern Convolutional Neural Networks (CNNs).
    Unlike standard pixel analysis, these networks extract *semantic features*.
    
    Mathematically, for a patch $x$, the network outputs a high-dimensional vector $z$:
    """)
    st.latex(r"z = f_{\theta}(x) \in \mathbb{R}^{1024}")
    st.markdown("""
    This vector $z$ is invariant to small rotations and noise, capturing the "essence" (texture, shape) of the patch.
    """)

    st.divider()

    st.header("2. Dimensionality Reduction (PCA)")
    st.markdown("""
    The raw feature vectors are too large (e.g., 1024 dimensions) and contain redundant information. 
    We use **Principal Component Analysis (PCA)** to project them into a lower-dimensional space (e.g., 50 dimensions) 
    that preserves the maximum variance.
    
    We find a projection matrix $W$ by solving the eigenvalue problem for the covariance matrix $C$:
    """)
    st.latex(r"C = \frac{1}{n} \sum_{i=1}^{n} (z_i - \bar{z})(z_i - \bar{z})^T")
    st.markdown("""
    We select the top $k$ eigenvectors corresponding to the largest eigenvalues. This compresses the data while keeping the signal.
    """)

    st.divider()

    st.header("3. Unsupervised Anomaly Detection (Isolation Forest)")
    st.markdown("""
    How do we know what is "interesting" without any labels? We assume that **rare** things are interesting.
    We use an **Isolation Forest**, which builds random decision trees.
    
    * **Common patches (Background):** Require many splits to isolate (Deep in the tree).
    * **Rare patches (Defects):** Require few splits to isolate (Shallow in the tree).
    
    The anomaly score is defined as:
    """)
    st.latex(r"s(x, n) = 2^{- \frac{E(h(x))}{c(n)}}")
    st.markdown("""
    * $h(x)$: Path length to isolate sample $x$.
    * $c(n)$: Average path length of a binary search tree (Normalization).
    
    Scores close to 1 indicate high anomaly (high priority for scanning).
    """)

    st.divider()

    st.header("4. Active Learning (Teacher Mode)")
    st.markdown("""
    When you click "Find More Like This", you turn the system into a **Supervised** learner. 
    We train a **Logistic Regression** classifier on the fly using your selected patch as a *Positive* example ($y=1$) 
    and random background patches as *Negative* examples ($y=0$).
    
    The model learns weights $w$ to maximize the likelihood:
    """)
    st.latex(r"P(y=1|z) = \frac{1}{1 + e^{-(w^T z + b)}}")
    st.markdown("""
    This probability map becomes the new scanning priority, effectively "teaching" the microscope what you care about in milliseconds.
    """)
