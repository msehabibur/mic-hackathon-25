#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py: The main Streamlit user interface for the DeepScan Pro application.
This script handles the UI layout, user inputs, visualization, and hardware simulation.
"""
import io
import json
import time
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
from skimage import measure

# --- Local Module Imports ---
from config import PAGE_CONFIG, AVAILABLE_MODELS, DEFAULT_PATCH_SIZES, DEFAULT_PCA_DIM, HARDWARE_NAME
from utils import load_image_grayscale, get_default_image
from ai_core import run_analysis_pipeline, train_classifier, search_by_text

# =====================================================================================
# Main Streamlit Application UI
# =====================================================================================
def main():
    st.set_page_config(**PAGE_CONFIG)

    # --- Initialize Session State ---
    if "results" not in st.session_state: st.session_state.results = None
    if "img_cache" not in st.session_state: st.session_state.img_cache = None
    if "history" not in st.session_state: st.session_state.history = []
    if "connected" not in st.session_state: st.session_state.connected = False
    if "current_score" not in st.session_state: st.session_state.current_score = None
    if "current_map" not in st.session_state: st.session_state.current_map = None
    if "mode" not in st.session_state: st.session_state.mode = "Unsupervised"

    st.sidebar.title("🧬 Navigator")
    page_selection = st.sidebar.radio("Go to:", ["🚀 Main Application", "📘 The Math Behind It"])

    # =================================================================================
    # PAGE 1: Main Application
    # =================================================================================
    if page_selection == "🚀 Main Application":
        st.title("🔬 DeepScan Pro: Intelligent Microscopy")
        
        # --- Sidebar: Hardware & Inputs ---
        with st.sidebar:
            st.divider()
            st.subheader("Hardware Status")
            if not st.session_state.connected:
                st.error("🔴 Disconnected")
                if st.button("🔌 Connect Microscope"):
                    with st.spinner("Handshaking with Controller..."):
                        time.sleep(1.0)
                    st.session_state.connected = True
                    st.rerun()
            else:
                st.success(f"🟢 Connected ({HARDWARE_NAME})")
                if st.button("❌ Disconnect"):
                    st.session_state.connected = False
                    st.rerun()
            st.divider()

            st.header("1. Input & Settings")
            uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "tif"])
            
            # Load Image Logic
            if uploaded:
                img_bytes = uploaded.getvalue()
                img = load_image_grayscale(io.BytesIO(img_bytes))
                st.success("✅ Custom Image Loaded")
            elif st.session_state.img_cache is not None:
                img = st.session_state.img_cache 
            else:
                img = get_default_image()
                st.session_state.img_cache = img
                st.info("ℹ️ Loaded Default STEM_example")

            # Update Cache
            if st.session_state.img_cache is None or not np.array_equal(img, st.session_state.img_cache):
                st.session_state.img_cache = img
                st.session_state.results = None
                st.session_state.history = []

            # Model Settings
            backbone = st.selectbox("Vision Backbone", AVAILABLE_MODELS, index=1)
            
            with st.expander("📏 Calibration (Physical Units)"):
                nm_per_pixel = st.number_input("Scale (nm per pixel)", value=1.0, min_value=0.01, format="%.2f")
                st.caption(f"100 pixels = {100*nm_per_pixel:.1f} nm")
                
            with st.expander("🛠️ Advanced Config"):
                patch_size_input = st.multiselect("Patch Sizes", [32, 64, 128], default=DEFAULT_PATCH_SIZES)
                stride_val = st.slider("Overlap (Stride divisor)", 1, 4, 2)
                pca_dim = st.slider("PCA Dim", 10, 100, DEFAULT_PCA_DIM)

            run_btn = st.button("🚀 Run Analysis")
            if st.button("Reset 🗑️"):
                st.session_state.clear()
                st.rerun()

        # --- Logic: Run Analysis ---
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

        # --- Logic: Display Results ---
        if st.session_state.results is not None:
            res = st.session_state.results
            score = st.session_state.current_score
            
            top_idx = np.argsort(score)[-10:][::-1]
            top_regions = [{"rank": r+1, "id": i, "i": res["coords"][i][0], "j": res["coords"][i][1], "size": res["coords"][i][2]} for r, i in enumerate(top_idx)]

            tab1, tab2, tab3, tab4 = st.tabs(["👁️ Visual Scan", "📏 Quantification", "💬 Text Search", "📊 Efficiency"])

            # --- TAB 1: Visual ---
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
                    
                    # Draw Boxes and Annotations (FIXED: Added ax.text back)
                    for r in top_regions:
                        rect = mpatches.Rectangle((r["j"], r["i"]), r["size"], r["size"], linewidth=2, edgecolor="lime", facecolor="none")
                        ax.add_patch(rect)
                        # THIS IS THE FIX YOU ASKED FOR:
                        ax.text(r["j"], max(0, r["i"]-5), str(r["rank"]), color="lime", fontsize=12, weight="bold")
                    
                    ax.legend(loc="lower right")
                    ax.axis("off")
                    st.pyplot(fig)
                    
                    # Protocol Download
                    protocol_data = {
                        "timestamp": "2023-10-27T10:00:00",
                        "scale_nm": nm_per_pixel,
                        "scan_points": [{"x": int(r['j']), "y": int(r['i']), "priority": int(r['rank'])} for r in top_regions]
                    }
                    if st.download_button("💾 Download Scan Protocol (.json)", json.dumps(protocol_data, indent=2), "scan_protocol.json", "application/json"):
                        st.balloons()
                        st.success("Protocol exported!")

                with c2:
                    st.subheader(f"Heatmap ({st.session_state.mode})")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(img, cmap="gray", alpha=0.4)
                    im = ax.imshow(st.session_state.current_map, cmap="jet", alpha=0.6)
                    ax.axis("off")
                    plt.colorbar(im, ax=ax)
                    st.pyplot(fig)

                # --- Defect Gallery ---
                st.divider()
                st.subheader("🔍 Identified Anomalies (Zoomed Views)")
                cols = st.columns(5)
                for idx, col in enumerate(cols):
                    if idx < 5 and idx < len(top_regions):
                        r = top_regions[idx]
                        patch_img = img[r['i']:r['i']+r['size'], r['j']:r['j']+r['size']]
                        col.image(patch_img, caption=f"Rank #{r['rank']}", use_column_width=True, clamp=True)
                st.divider()

                # --- Interactive Feedback ---
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

            # --- TAB 2: Quantification ---
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

            # --- TAB 3: Text Search ---
            with tab3:
                st.markdown("### 🗣️ Semantic Search")
                st.info("Type a query to search the image using Multimodal CLIP AI.")
                query = st.text_input("Query:", placeholder="e.g. 'dark circular defects' or 'linear cracks'")
                if st.button("Search 🔎"):
                    with st.spinner("CLIP is seeing the image..."):
                        text_scores, text_map = search_by_text(res["raw_patches"], query, img.shape, res["coords"])
                        st.session_state.current_score = text_scores
                        st.session_state.current_map = text_map
                        st.session_state.mode = f"Text: '{query}'"
                        st.session_state.history.append(text_map)
                        st.rerun()

            # --- TAB 4: Efficiency ---
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
                st.download_button("Download CSV 📄", df_rep.to_csv(index=False), "efficiency_report.csv", "text/csv")

    # =================================================================================
    # PAGE 2: The Math Behind It
    # =================================================================================
    elif page_selection == "📘 The Math Behind It":
        st.title("📘 The Science of DeepScan")
        st.markdown("""
        This platform transforms a microscope into an intelligent agent using a pipeline of **Computer Vision**, 
        **Dimensionality Reduction**, and **Active Learning**.
        """)
        
        st.divider()
        st.header("1. Sliding Window & Feature Extraction")
        st.markdown("The high-resolution microscope image $I$ is essentially a massive matrix of pixels. We cannot process it all at once, so we decompose it into small overlapping patches.")
        st.latex(r"z = f_{\theta}(x) \in \mathbb{R}^{1024}")
        st.markdown("This vector $z$ is invariant to small rotations and noise, capturing the 'essence' of the patch.")

        st.divider()
        st.header("2. Dimensionality Reduction (PCA)")
        st.markdown("We find a projection matrix $W$ by solving the eigenvalue problem for the covariance matrix $C$:")
        st.latex(r"C = \frac{1}{n} \sum_{i=1}^{n} (z_i - \bar{z})(z_i - \bar{z})^T")

        st.divider()
        st.header("3. Unsupervised Anomaly Detection (Isolation Forest)")
        st.markdown("The anomaly score $s(x, n)$ for a sample $x$ is defined as:")
        st.latex(r"s(x, n) = 2^{- \frac{E(h(x))}{c(n)}}")
        st.markdown("* $h(x)$: Path length to isolate sample $x$. \n * $c(n)$: Normalization factor.")

        st.divider()
        st.header("4. Active Learning (Teacher Mode)")
        st.markdown("The model learns weights $w$ to maximize the likelihood:")
        st.latex(r"P(y=1|z) = \frac{1}{1 + e^{-(w^T z + b)}}")

if __name__ == "__main__":
    main()
