import io
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import pandas as pd
import numpy as np

# Import from the utils folder
from utils.al_core import load_image_grayscale, run_initial_analysis, train_active_learner, compute_top_k

st.title("🚀 Active Discovery Engine")

# --- SIDEBAR: Configuration ---
st.sidebar.header("1. Input & Settings")
uploaded = st.sidebar.file_uploader("Upload SEM/STEM Image", type=["png", "jpg", "jpeg", "tif"])
backbone = st.sidebar.selectbox("Backbone", ["resnet18", "resnet50", "convnext_tiny"])

if uploaded:
    img_bytes = uploaded.getvalue()
    img = load_image_grayscale(io.BytesIO(img_bytes))
    
    # Check if this is a new image
    if st.session_state.img_cache is None or not np.array_equal(img, st.session_state.img_cache):
        st.session_state.img_cache = img
        st.session_state.results = None # Reset results
        st.session_state.history = []

    # Run Button
    if st.sidebar.button("Run Initial Scan ⚡"):
        with st.spinner("Extracting features and detecting anomalies..."):
            res = run_initial_analysis(img, backbone=backbone)
            st.session_state.results = res
            st.session_state.current_score = res["score"]
            st.session_state.current_map = res["scan_map"]
            st.session_state.mode = "Unsupervised (Anomaly)"
            # Save initial state to history
            st.session_state.history.append({"step": 0, "type": "Unsupervised", "map": res["scan_map"]})

# --- MAIN INTERFACE ---
if st.session_state.results is not None:
    res = st.session_state.results
    img = st.session_state.img_cache
    score = st.session_state.current_score
    scan_map = st.session_state.current_map
    
    # Layout: Top Row (Heatmaps)
    c1, c2 = st.columns(2)
    
    top_regions = compute_top_k(score, res["coords"], k=10)
    
    with c1:
        st.subheader("Original Image")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, cmap="gray")
        # Draw boxes
        for r in top_regions:
            rect = mpatches.Rectangle((r["j"], r["i"]), r["size"], r["size"], 
                                      linewidth=2, edgecolor="#00FF00", facecolor="none")
            ax.add_patch(rect)
            ax.text(r["j"], r["i"]-5, f"#{r['rank']}", color="#00FF00", fontsize=9, weight="bold")
        ax.axis("off")
        st.pyplot(fig)
        
    with c2:
        st.subheader(f"AI Attention Map ({st.session_state.mode})")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, cmap="gray", alpha=0.5)
        im = ax.imshow(scan_map, cmap="jet", alpha=0.6)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

    st.divider()

    # Layout: Teacher Mode (Bottom)
    st.markdown("### 👨‍🏫 Teacher Mode (Refine the AI)")
    
    col_teach_1, col_teach_2 = st.columns([1, 2])
    
    with col_teach_1:
        st.info("Train a custom classifier instantly.")
        
        # 1. Select Positive
        selected_rank = st.selectbox("Select Target Region (Positive)", [r["rank"] for r in top_regions])
        target = next(r for r in top_regions if r["rank"] == selected_rank)
        
        # 2. Select Negatives (Optional for advanced users, here simplified)
        st.write(f"**Target:** Region #{selected_rank} (Score: {target['score']:.2f})")
        
        if st.button("Train on this Region 🧠"):
            new_probs, new_map = train_active_learner(
                res["features"], res["coords"], img.shape, 
                positive_idx=target["id"]
            )
            # Update State
            st.session_state.current_score = new_probs
            st.session_state.current_map = new_map
            st.session_state.mode = f"Supervised (Target #{selected_rank})"
            
            # Log to history for the Efficiency Page
            step_n = len(st.session_state.history)
            st.session_state.history.append({"step": step_n, "type": "Supervised", "map": new_map})
            st.rerun()

        if st.button("Reset 🔄"):
            st.session_state.current_score = res["score"]
            st.session_state.current_map = res["scan_map"]
            st.session_state.mode = "Unsupervised"
            st.rerun()

    with col_teach_2:
        # Plotly Scatter
        df = pd.DataFrame(res["embedding"], columns=["x", "y"])
        df["score"] = score
        # Highlight selected
        df["size"] = 2
        df.loc[target["id"], "size"] = 15
        
        fig = px.scatter(df, x="x", y="y", color="score", size="size", 
                         color_continuous_scale="Jet", title="Feature Space Navigation")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Please upload an image to begin.")
