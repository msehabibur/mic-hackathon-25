import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import plotly.express as px

# Import core logic
from al_core import (
    load_image_grayscale, run_active_learning, compute_top_k, 
    rerank_global, train_interactive_model, generate_sam_masks
)

st.set_page_config(page_title="Microscopy Intelligent Discovery", layout="wide", page_icon="🔬")

# --- SESSION STATE INITIALIZATION ---
if "results" not in st.session_state: st.session_state.results = None
if "img_cache" not in st.session_state: st.session_state.img_cache = None
if "img_pil" not in st.session_state: st.session_state.img_pil = None
# Labels for classifier: sets of patch IDs
if "pos_labels" not in st.session_state: st.session_state.pos_labels = set()
if "neg_labels" not in st.session_state: st.session_state.neg_labels = set()
if "sam_masks" not in st.session_state: st.session_state.sam_masks = None


st.title("🔬 Next-Best-Scan: Intelligent Discovery")

# -----------------------
# SIDEBAR: Setup & Labeling
# -----------------------
st.sidebar.header("1. Setup")
uploaded = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "tif"])
backbone = st.sidebar.selectbox("Backbone", ["resnet18", "convnext_tiny"], index=0)

if uploaded:
    img_bytes = uploaded.getvalue()
    img, img_pil = load_image_grayscale(io.BytesIO(img_bytes))
    # Reset state on new upload
    if st.session_state.img_cache is None or not np.array_equal(img, st.session_state.img_cache):
         st.session_state.img_cache = img
         st.session_state.img_pil = img_pil
         st.session_state.results = None
         st.session_state.pos_labels = set()
         st.session_state.neg_labels = set()
         st.session_state.sam_masks = None

    if st.sidebar.button("🚀 Run Initial Analysis"):
        with st.spinner("Running unsupervised analysis & preparing SAM..."):
            res = run_active_learning(img, backbone=backbone)
            st.session_state.results = res
            st.session_state.current_score = res["score"]
            st.session_state.current_map = res["scan_map"]
            st.session_state.mode = "Unsupervised (Isolation Forest)"
            
            # Pre-compute SAM masks for top regions
            top_regs_initial = compute_top_k(res["score"], res["coords"], k=5)
            st.session_state.sam_masks = generate_sam_masks(img_pil, top_regs_initial)
            st.rerun()

# --- LABELING STATION (Only active after run) ---
if st.session_state.results is not None:
    st.sidebar.divider()
    st.sidebar.header("2. Labeling Station")
    st.sidebar.info("Select points on UMAP plot, then assign labels here.")
    
    # Get selections from Plotly chart
    selection_data = st.session_state.get("umap_selection", {}).get("selection", {}).get("points", [])
    selected_ids = [p["customdata"][0] for p in selection_data] if selection_data else []
    
    st.sidebar.write(f"**Selected points:** {len(selected_ids)}")

    b1, b2 = st.sidebar.columns(2)
    if b1.button("➕ Add to Positive"):
        st.session_state.pos_labels.update(selected_ids)
    if b2.button("➖ Add to Negative"):
        st.session_state.neg_labels.update(selected_ids)
        
    st.sidebar.write(f"Pos Labels: {len(st.session_state.pos_labels)}")
    st.sidebar.write(f"Neg Labels: {len(st.session_state.neg_labels)}")
    
    if st.sidebar.button("🧹 Clear All Labels"):
        st.session_state.pos_labels = set()
        st.session_state.neg_labels = set()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.header("3. Train & Refine")
    if st.sidebar.button("🧠 Train Classifier"):
        if len(st.session_state.pos_labels) < 2 or len(st.session_state.neg_labels) < 2:
            st.sidebar.error("Need at least 2 Positive and 2 Negative labels.")
        else:
            with st.spinner("Training interactive classifier..."):
                res = st.session_state.results
                probs, map_ = train_interactive_model(
                    res["features"], res["coords"], img.shape,
                    list(st.session_state.pos_labels),
                    list(st.session_state.neg_labels)
                )
                st.session_state.current_score = probs
                st.session_state.current_map = map_
                st.session_state.mode = "Interactive Classifier (Trained)"
                st.rerun()

    if st.sidebar.button("🔄 Reset to Unsupervised"):
        st.session_state.current_score = st.session_state.results["score"]
        st.session_state.current_map = st.session_state.results["scan_map"]
        st.session_state.mode = "Unsupervised (Isolation Forest)"
        st.rerun()


# -----------------------
# MAIN DASHBOARD
# -----------------------
if st.session_state.results is not None:
    res = st.session_state.results
    img = st.session_state.img_cache
    score = st.session_state.current_score
    scan_map = st.session_state.current_map
    
    # Compute Top-K for current scores
    top_regions = compute_top_k(score, res["coords"], k=5)

    st.metric("Current Mode", st.session_state.mode)

    # --- ROW 1: VISUALIZATION ---
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("Top-5 SAM Segmentation")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        
        # Overlay SAM masks if available (only for unsupervised mode for now)
        if st.session_state.mode.startswith("Unsupervised") and st.session_state.sam_masks:
            # Create a colormap: transparent for 0, color for 1
            colors = ['lime', 'cyan', 'magenta', 'yellow', 'red']
            
            for i, mask in enumerate(st.session_state.sam_masks):
                if mask is not None and i < len(top_regions):
                    # Create custom colormap for this mask
                    cmap = ListedColormap(['none', colors[i%len(colors)]])
                    ax.imshow(mask, cmap=cmap, alpha=0.5)
                    
                    # Add text label near the center of the region
                    r = top_regions[i]
                    ax.text(r["j"], max(r["i"]-5, 0), f"#{r['rank']}", color=colors[i%len(colors)], weight='bold', bbox=dict(facecolor='black', alpha=0.5))
        else:
            # Fallback to boxes if SAM failed or in classifier mode
            for r in top_regions:
                rect = mpatches.Rectangle((r["j"], r["i"]), r["size"], r["size"], linewidth=2, edgecolor="lime", facecolor="none")
                ax.add_patch(rect)
                ax.text(r["j"], r["i"], str(r["rank"]), color="lime", weight="bold")

        st.pyplot(fig)

    with c2:
        st.subheader("Priority Heatmap")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, cmap="gray", alpha=0.4)
        im = ax.imshow(scan_map, cmap="inferno", alpha=0.7)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

    st.divider()

    # --- ROW 2: INTERACTIVE UMAP ---
    st.subheader("Interactive Feature Space (Select points to label)")
    
    # Prepare Data for Plotly
    df_umap = pd.DataFrame(res["embedding"], columns=["x", "y"])
    df_umap["score"] = score
    df_umap["id"] = range(len(score))
    
    # Define colors based on labels
    df_umap["color"] = "Unlabeled"
    df_umap.loc[list(st.session_state.pos_labels), "color"] = "Positive"
    df_umap.loc[list(st.session_state.neg_labels), "color"] = "Negative"
    
    color_map = {"Unlabeled": "lightgrey", "Positive": "lime", "Negative": "red"}

    fig = px.scatter(
        df_umap, x="x", y="y", 
        color="color",
        color_discrete_map=color_map,
        hover_data=["id", "score"],
        custom_data=["id"], # Crucial for selection
        title="UMAP (Drag to select multiple points)",
    )
    
    # Enable multiple selection box
    fig.update_layout(dragmode='select', clickmode='event+select')
    fig.update_traces(selected=dict(marker=dict(opacity=1.0, line=dict(width=2, color='black'))),
                      unselected=dict(marker=dict(opacity=0.3)))

    # Display chart and capture selection state
    st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="umap_selection")
    st.info("Tip: Use the Box Select tool in the Plotly toolbar to select many points at once.")

elif not uploaded:
    st.info("👈 Please upload an image in the sidebar to start.")
