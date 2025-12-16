import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import your core logic
from al_core import load_image_grayscale, run_active_learning

st.set_page_config(page_title="Microscopy-Informed Active Learning", layout="wide")

st.title("🔬 Microscopy-Informed Active Learning: Next-Best-Scan")

# -----------------------
# Sidebar controls (Global)
# -----------------------
st.sidebar.header("Settings")

backbone = st.sidebar.selectbox("Backbone", ["resnet18", "convnext_tiny", "vit_base_patch16_224"], index=0)
top_k = st.sidebar.slider("Top-K regions", 1, 50, 10)

patch_sizes = st.sidebar.multiselect("Patch sizes", [16, 32, 64, 96, 128], default=[32, 64])
stride_map = {16: 8, 32: 16, 64: 32, 96: 48, 128: 64}
strides = [stride_map[p] for p in patch_sizes]

pca_dim = st.sidebar.slider("PCA dim", 10, 256, 50)
umap_neighbors = st.sidebar.slider("UMAP neighbors", 5, 50, 15)
umap_min_dist = st.sidebar.slider("UMAP min_dist", 0.0, 0.9, 0.1)

alpha = st.sidebar.slider("Heatmap alpha", 0.1, 1.0, 0.6)

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Upload SEM/STEM image", type=["png", "jpg", "jpeg", "tif", "tiff"])

run_btn = st.sidebar.button("🚀 Run Next-Best-Scan")

# -----------------------
# Tabs Setup
# -----------------------
tab_main, tab_info = st.tabs(["🚀 Main Application", "ℹ️ How It Works"])

# ==========================================
# TAB 1: Main Application Logic
# ==========================================
with tab_main:
    if uploaded is None:
        st.info("Upload an image in the sidebar to begin.")
    else:
        # Load and display input
        img = load_image_grayscale(uploaded)
        H, W = img.shape

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input Image")
            fig = plt.figure(figsize=(6, 6))
            plt.imshow(img, cmap="gray")
            plt.axis("off")
            st.pyplot(fig)

        with col2:
            st.subheader("Image Info")
            st.write(f"Shape: {H} × {W}")
            st.write(f"Min/Max: {float(img.min()):.3f} / {float(img.max()):.3f}")

        # Run Logic
        if run_btn:
            @st.cache_resource(show_spinner=False)
            def _cached_model_run(img_bytes, backbone, patch_sizes, strides, top_k, pca_dim, umap_neighbors, umap_min_dist):
                img_buf = io.BytesIO(img_bytes)
                img_local = load_image_grayscale(img_buf)
                return run_active_learning(
                    img_local,
                    backbone=backbone,
                    patch_sizes=tuple(patch_sizes),
                    strides=tuple(strides),
                    top_k=top_k,
                    pca_dim=pca_dim,
                    umap_neighbors=umap_neighbors,
                    umap_min_dist=umap_min_dist,
                )

            with st.spinner("Computing embeddings + scan map..."):
                img_bytes = uploaded.getvalue()
                out = _cached_model_run(img_bytes, backbone, patch_sizes, strides, top_k, pca_dim, umap_neighbors, umap_min_dist)

            scan_map = out["scan_map"]
            embedding = out["embedding"]
            score = out["score"]
            top_regions = out["top_regions"]

            st.success(f"Done. Device: **{out['device']}** | Features: **{out['features_shape']}**")

            # Visualization
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🔥 Next-Best Scan Heatmap")
                fig = plt.figure(figsize=(6, 6))
                plt.imshow(img, cmap="gray")
                plt.imshow(scan_map, cmap="jet", alpha=alpha)
                plt.axis("off")
                st.pyplot(fig)

            with c2:
                st.subheader("✅ Top-K Overlay")
                fig = plt.figure(figsize=(6, 6))
                plt.imshow(img, cmap="gray")
                ax = plt.gca()
                ax.axis("off")
                for r in top_regions:
                    i, j, size = r["i"], r["j"], r["size"]
                    rect = mpatches.Rectangle((j, i), size, size, linewidth=2, edgecolor="lime", facecolor="none")
                    ax.add_patch(rect)
                    ax.text(j, max(i - 5, 0), str(r["rank"]),
                            color="yellow", fontsize=12, weight="bold",
                            bbox=dict(facecolor="black", alpha=0.6, pad=2))
                st.pyplot(fig)

            st.subheader("Patch Representation Space (UMAP)")
            fig = plt.figure(figsize=(7, 5))
            plt.scatter(embedding[:, 0], embedding[:, 1], c=score, cmap="inferno", s=6)
            plt.colorbar(label="Informativeness")
            plt.title("UMAP of Patch Embeddings")
            st.pyplot(fig)

            st.subheader("🔥 Top Regions Table + Download")
            df = pd.DataFrame(top_regions)
            st.dataframe(df, use_container_width=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Top-K CSV", data=csv_bytes, file_name="top_regions.csv", mime="text/csv")
        
        elif uploaded is not None:
             st.warning("Click **Run Next-Best-Scan** to compute recommendations.")

# ==========================================
# TAB 2: How It Works (Educational)
# ==========================================
with tab_info:
    st.header("How the Active Learning Pipeline Works")
    
    st.markdown("""
    This tool uses **Unsupervised Active Learning** to find scientifically interesting regions in microscopy images. 
    It assumes that "interesting" regions are those that are statistically rare, complex, or distinct from the background.
    """)

    st.divider()

    # Section 1: Extraction
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("1. Multi-Scale Patch Extraction")
        st.markdown("""
        We slice the image into small squares (patches).
        * **Multi-Scale:** We use different sizes (e.g., 32px for atomic detail, 64px for grain boundaries).
        * **Sliding Window:** Patches overlap to ensure we don't miss features at the edges.
        """)
    with col_b:
        st.subheader("2. Deep Feature Encoding")
        st.markdown("""
        Each patch is processed by a neural network (Backbone).
        * **Model:** Uses pre-trained networks like **ResNet** or **ViT**.
        * **Embedding:** Converts the image patch into a vector of numbers (features) representing texture, edges, and shapes.
        """)

    st.divider()

    # Section 2: Scoring
    st.subheader("3. Informativeness Scoring")
    st.info("We calculate a score for every patch to determine how 'scan-worthy' it is.")

    st.markdown(r"""
    The final score is a composite of three metrics:
    
    $$
    \text{Score} = \text{Novelty} + \text{Uncertainty} + \text{Diversity}
    $$
    
    | Metric | Method | Interpretation |
    | :--- | :--- | :--- |
    | **Novelty** | **Mahalanobis Distance** | Measures how far a patch is from the statistical "center" of the data distribution. Finds outliers. |
    | **Uncertainty** | **Distance from Mean** | Simple distance from the average patch. Highlights regions distinct from the dominant background. |
    | **Diversity** | **Feature Magnitude (L2)** | Measures the "strength" of the features. Complex textures yield higher magnitudes than flat backgrounds. |
    """)

    st.divider()

    # Section 3: Visualization
    st.subheader("4. Visualization")
    st.markdown("""
    * **Heatmap:** We project the scores back onto the image. Overlapping patches vote on the pixel importance.
    * **UMAP:** A technique to visualize high-dimensional data in 2D. 
        * **Clusters:** Similar patches (e.g., vacuum/background) group together.
        * **Outliers:** Rare defects often appear as isolated points.
    """)
