import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px

# Import core logic
from al_core import load_image_grayscale, run_active_learning, compute_top_k, rerank_global

st.set_page_config(page_title="Microscopy Active Learning", layout="wide", page_icon="🔬")

# Initialize Session State
if "results" not in st.session_state:
    st.session_state.results = None
if "img_cache" not in st.session_state:
    st.session_state.img_cache = None
if "mode" not in st.session_state:
    st.session_state.mode = "Unsupervised"

st.title("🔬 Next-Best-Scan: Interactive Discovery")

# -----------------------
# Sidebar
# -----------------------
st.sidebar.header("1. Upload & Settings")
uploaded = st.sidebar.file_uploader("Upload SEM/STEM Image", type=["png", "jpg", "jpeg", "tif"])
backbone = st.sidebar.selectbox("Model Backbone", ["resnet18", "resnet50", "convnext_tiny"], index=0)

with st.sidebar.expander("Advanced Params"):
    patch_sizes = st.multiselect("Patch Sizes", [32, 64, 128], default=[32, 64])
    stride_map = {32: 16, 64: 32, 128: 64}
    strides = [stride_map[p] for p in patch_sizes]
    pca_dim = st.slider("PCA Dim", 20, 100, 50)

run_btn = st.sidebar.button("🚀 Run Analysis")

# -----------------------
# Main Execution
# -----------------------
if uploaded:
    # Load Image
    img_bytes = uploaded.getvalue()
    img = load_image_grayscale(io.BytesIO(img_bytes))
    st.session_state.img_cache = img

    # Run Pipeline
    if run_btn:
        with st.spinner("Analyzing textures and anomalies..."):
            res = run_active_learning(
                img, 
                backbone=backbone,
                patch_sizes=tuple(patch_sizes),
                strides=tuple(strides),
                pca_dim=pca_dim
            )
            st.session_state.results = res
            st.session_state.current_score = res["score"]
            st.session_state.current_map = res["scan_map"]
            st.session_state.mode = "Unsupervised"

# -----------------------
# Tabs Layout
# -----------------------
tab_main, tab_info = st.tabs(["🚀 Main Application", "ℹ️ How It Works"])

with tab_main:
    if st.session_state.results is not None:
        res = st.session_state.results
        img = st.session_state.img_cache
        score = st.session_state.current_score
        scan_map = st.session_state.current_map
        
        top_regions = compute_top_k(score, res["coords"], k=10)

        # --- Header Metrics ---
        m1, m2, m3 = st.columns(3)
        m1.metric("Mode", st.session_state.mode)
        m2.metric("Patches Analyzed", len(score))
        m3.metric("Backend", res["device"])

        st.divider()

        # --- ROW 1: Heatmaps ---
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.subheader("Input Image")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            
            for r in top_regions:
                rect = mpatches.Rectangle((r["j"], r["i"]), r["size"], r["size"], 
                                        linewidth=2, edgecolor="lime", facecolor="none")
                ax.add_patch(rect)
                ax.text(r["j"], max(r["i"]-5, 0), str(r["rank"]), color="lime", fontsize=10, weight="bold")
            st.pyplot(fig)

        with c2:
            st.subheader("Priority Heatmap")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img, cmap="gray", alpha=0.4)
            im = ax.imshow(scan_map, cmap="jet", alpha=0.6)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)

        st.divider()

        # --- ROW 2: Human-in-the-Loop Feedback ---
        st.header("🧠 Teacher Mode (Human-in-the-Loop)")
        
        fb_col1, fb_col2 = st.columns([1, 2])
        
        with fb_col1:
            st.info("Select a region that looks interesting to find more like it.")
            
            # Dropdown to select a region from Top-10
            selected_rank = st.selectbox(
                "Select a ROI (Rank):", 
                options=[r["rank"] for r in top_regions],
                format_func=lambda x: f"Region #{x}"
            )
            
            # Get the actual index in the feature array
            selection = next(r for r in top_regions if r["rank"] == selected_rank)
            sel_idx = selection["id"]
            
            c_sim, c_diff = st.columns(2)
            if c_sim.button("Find Similar 🔍"):
                new_scores, new_map = rerank_global(res["features"], res["coords"], img.shape, sel_idx, mode="similar")
                st.session_state.current_score = new_scores
                st.session_state.current_map = new_map
                st.session_state.mode = f"Similar to #{selected_rank}"
                st.rerun()
                
            if c_diff.button("Find Anomaly ⚡"):
                new_scores, new_map = rerank_global(res["features"], res["coords"], img.shape, sel_idx, mode="dissimilar")
                st.session_state.current_score = new_scores
                st.session_state.current_map = new_map
                st.session_state.mode = f"Dissimilar to #{selected_rank}"
                st.rerun()

            if st.button("Reset to Default 🔄"):
                st.session_state.current_score = res["score"]
                st.session_state.current_map = res["scan_map"]
                st.session_state.mode = "Unsupervised"
                st.rerun()

        with fb_col2:
            st.subheader("Patch Similarity Space (Interactive)")
            # Prepare Data for Plotly
            df_umap = pd.DataFrame(res["embedding"], columns=["x", "y"])
            df_umap["score"] = st.session_state.current_score
            
            # Highlight selected point
            df_umap["size"] = 2
            df_umap.loc[sel_idx, "size"] = 15 # Make selected point big
            df_umap["color_type"] = "Normal"
            df_umap.loc[sel_idx, "color_type"] = "Selected"

            fig = px.scatter(
                df_umap, x="x", y="y", 
                color="score", 
                size="size",
                color_continuous_scale="Jet",
                title="UMAP Feature Space",
                hover_data={"x": False, "y": False, "score": True}
            )
            fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

    elif not uploaded:
        st.info("👈 Please upload an image in the sidebar to start.")

with tab_info:
    st.header("🧠 The Logic Behind the Tool")
    
    st.subheader("1. The Backbone (Vision)")
    st.markdown("""
    We treat the microscope image not as pixels, but as a collection of concepts.
    * **Sliding Window:** We cut the image into small patches (e.g., 32x32). 
    * **Neural Network:** A pre-trained AI (ResNet/ConvNext) looks at each patch and converts it into a "fingerprint" (embedding). 
    """)
    
    st.subheader("2. Anomaly Detection (Isolation Forest)")
    st.markdown("""
    How do we know what is "interesting"?
    * We use an **Isolation Forest** algorithm on the fingerprints.
    * It builds random decision trees to try and isolate every patch.
    * **Rare/Anomalous patches** are easy to isolate (short paths).
    * **Background patches** are hard to isolate (deep paths).
    """)
    
    st.subheader("3. Human-in-the-Loop (Teacher Mode)")
    st.markdown("""
    Unsupervised learning is just a guess. The **Teacher Mode** lets you correct it.
    * When you click **'Find Similar'**, we take the fingerprint of your selection.
    * We calculate the **Cosine Similarity** between that fingerprint and every other patch.
    * The heatmap updates to highlight *only* the regions that match your specific scientific query.
    """)
