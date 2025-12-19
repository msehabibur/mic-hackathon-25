#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py: The main Streamlit user interface for the DeepScan Pro application.
"""
import io
import os
import json
import time
from datetime import datetime
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
from skimage import measure

# Import configuration and modules
from config import PAGE_CONFIG, AVAILABLE_MODELS, DEFAULT_PATCH_SIZES, DEFAULT_PCA_DIM, HARDWARE_NAME
from utils import load_image_grayscale, get_default_image
from ai_core import run_analysis_pipeline, train_classifier, search_by_text, compute_fft_metrics

# =====================================================================================
# Helper: System Logger
# =====================================================================================
def log_message(msg):
    """Adds a timestamped message to the session logs."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    st.session_state.logs.append(entry)
    if len(st.session_state.logs) > 10:
        st.session_state.logs.pop(0)

# =====================================================================================
# Main Streamlit Application UI
# =====================================================================================
def main():
    """Main function to run the Streamlit application interface."""
    st.set_page_config(**PAGE_CONFIG)

    # --- Session State Initialization ---
    if "results" not in st.session_state: st.session_state.results = None
    if "img_cache" not in st.session_state: st.session_state.img_cache = None
    if "history" not in st.session_state: st.session_state.history = []
    if "connected" not in st.session_state: st.session_state.connected = False
    if "current_score" not in st.session_state: st.session_state.current_score = None
    if "current_map" not in st.session_state: st.session_state.current_map = None
    if "mode" not in st.session_state: st.session_state.mode = "Unsupervised"
    if "logs" not in st.session_state: st.session_state.logs = [] 

    st.title("🔬 DeepScan Pro v7")
    st.write("An Active Learning Pipeline for Intelligent Atomic Microscopy")

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("🔌 Hardware Status")
        if not st.session_state.connected:
            st.error("🔴 Disconnected")
            if st.button("Connect Microscope"):
                with st.spinner("Handshaking with Controller..."):
                    time.sleep(0.8)
                    log_message(f"Connected to {HARDWARE_NAME}")
                st.session_state.connected = True
                st.rerun()
        else:
            st.success(f"🟢 Connected ({HARDWARE_NAME})")
            if st.button("Disconnect"):
                log_message("Disconnected from hardware.")
                st.session_state.connected = False
                st.rerun()

        st.divider()
        st.subheader("📟 System Logs")
        log_text = "\n".join(st.session_state.logs)
        st.text_area("Console Output", value=log_text, height=150, disabled=True)
        st.divider()

        st.subheader("🖼️ Input Data")
        uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "tif"])
        
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

        if st.session_state.img_cache is None or not np.array_equal(img, st.session_state.img_cache):
            st.session_state.img_cache = img
            st.session_state.results = None
            st.session_state.history = []
            log_message("New image loaded into memory.")

        st.subheader("🤖 AI Model")
        backbone = st.selectbox("Vision Backbone", AVAILABLE_MODELS, index=1)

        st.subheader("📏 Calibration")
        nm_per_pixel = st.number_input("Scale (nm/px)", value=1.0, min_value=0.01, format="%.2f")

        st.subheader("🛠️ Parameters")
        patch_size_input = st.multiselect("Patch Sizes", [32, 64, 128], default=DEFAULT_PATCH_SIZES)
        stride_val = st.slider("Overlap (Stride divisor)", 1, 4, 2)
        pca_dim = st.slider("PCA Dim", 10, 100, DEFAULT_PCA_DIM)

        st.divider()
        if st.button("🔴 Reset Session"):
            st.session_state.clear()
            st.rerun()

    # --- Tabs ---
    tab_about, tab_run, tab_quant, tab_analytics = st.tabs(["💡 How It Works", "1️⃣ Visual Scan", "2️⃣ Quantification & Search", "📊 Analytics"])

    # --- TAB 1: How It Works ---
    with tab_about:
        st.header("How DeepScan Works: From Pixels to Protocols")
        with st.container(border=True):
            st.markdown("**Md Habibur Rahman** *School of Materials Engineering, Purdue University*")
        
        if os.path.exists("workflow.png"):
            st.image("workflow.png", caption="Figure 1: The DeepScan Pro Architecture.", use_container_width=True)
        else:
            st.warning("⚠️ workflow.png not found.")

        st.info("This application implements a multi-stage pipeline to turn passive images into active scanning protocols.")

    # --- TAB 2: Visual Scan ---
    with tab_run:
        st.header("🚀 AI-Powered Analysis")
        
        if st.button("✨ Run Full Analysis", type="primary"):
            log_message(f"Starting analysis with {backbone}...")
            with st.spinner(f"Analyzing with {backbone}..."):
                strides = tuple([p // stride_val for p in patch_size_input])
                res = run_analysis_pipeline(img, backbone, tuple(patch_size_input), strides, pca_dim=pca_dim)
                if res:
                    st.session_state.results = res
                    st.session_state.current_score = res["score"]
                    st.session_state.current_map = res["scan_map"]
                    st.session_state.mode = "Unsupervised"
                    st.session_state.history.append(res["scan_map"])
                    log_message("Analysis complete.")
                else:
                    st.error("Image too small.")

        if st.session_state.results is not None:
            res = st.session_state.results
            score = st.session_state.current_score
            
            top_idx = np.argsort(score)[-10:][::-1]
            top_regions = [{"rank": r+1, "id": i, "i": res["coords"][i][0], "j": res["coords"][i][1], "size": res["coords"][i][2]} for r, i in enumerate(top_idx)]

            # --- Simulation ---
            c_sim, c_view = st.columns([1, 3])
            with c_sim:
                st.markdown("### 🎮 Control")
                if st.button("▶️ Simulate Live Scan"):
                    log_message("Starting live scan simulation...")
                    placeholder = c_view.empty()
                    for step in range(1, len(top_regions) + 1):
                        current_regions = top_regions[:step]
                        
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.imshow(img, cmap="gray")
                        
                        # Draw Path
                        if len(current_regions) > 1:
                            py = [r["i"] + r["size"]//2 for r in current_regions]
                            px_coords = [r["j"] + r["size"]//2 for r in current_regions]
                            ax.plot(px_coords, py, 'r--', linewidth=2, alpha=0.8, label="Optimized Path")
                        
                        # Draw Boxes
                        last_r = current_regions[-1]
                        rect = mpatches.Rectangle((last_r["j"], last_r["i"]), last_r["size"], last_r["size"], linewidth=3, edgecolor="lime", facecolor="none")
                        ax.add_patch(rect)
                        
                        # Draw Text Label
                        ax.text(last_r["j"], max(0, last_r["i"]-5), str(last_r["rank"]), color="lime", fontsize=14, weight="bold")
                        
                        ax.axis("off")
                        ax.set_title(f"Scanning Region #{last_r['rank']}...", color="lime")
                        
                        placeholder.pyplot(fig)
                        plt.close(fig)
                        time.sleep(0.5)
                    
                    log_message("Scan simulation completed.")
                    st.rerun()

            # --- Static View (FIXED: Added Labels Back) ---
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Microscope View")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(img, cmap="gray")
                
                path_y = [r["i"] + r["size"]//2 for r in top_regions]
                path_x = [r["j"] + r["size"]//2 for r in top_regions]
                
                # FIXED: Added labels for the legend
                ax.plot(path_x, path_y, 'r--', linewidth=1.5, alpha=0.8, label="Optimized Path")
                ax.scatter(path_x[0], path_y[0], c='lime', s=100, zorder=5, label="Start Point")
                
                for r in top_regions:
                    rect = mpatches.Rectangle((r["j"], r["i"]), r["size"], r["size"], linewidth=2, edgecolor="lime", facecolor="none")
                    ax.add_patch(rect)
                    # FIXED: Ensure numbers appear
                    ax.text(r["j"], max(0, r["i"]-5), str(r["rank"]), color="lime", fontsize=12, weight="bold")
                
                ax.legend(loc="lower right")
                ax.axis("off")
                st.pyplot(fig)

            with col2:
                st.subheader(f"Priority Map ({st.session_state.mode})")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(img, cmap="gray", alpha=0.4)
                im = ax.imshow(st.session_state.current_map, cmap="jet", alpha=0.6)
                ax.axis("off")
                plt.colorbar(im, ax=ax)
                st.pyplot(fig)

            st.divider()
            
            # Defect Gallery
            st.subheader("🔍 Identified Anomalies")
            cols = st.columns(5)
            for idx, col in enumerate(cols):
                if idx < 5 and idx < len(top_regions):
                    r = top_regions[idx]
                    patch_img = img[r['i']:r['i']+r['size'], r['j']:r['j']+r['size']]
                    col.image(patch_img, caption=f"Rank #{r['rank']}", use_column_width=True, clamp=True)
            
            st.divider()

            # Teacher Mode
            c_teach, c_plot = st.columns([1, 2])
            with c_teach:
                st.subheader("👨‍🏫 Teacher Mode")
                sel_rank = st.selectbox("Select Interesting Region:", [r["rank"] for r in top_regions])
                target = next(r for r in top_regions if r["rank"] == sel_rank)
                
                # FFT Physics Check
                target_patch = img[target['i']:target['i']+target['size'], target['j']:target['j']+target['size']]
                fft_mag, crystal_score = compute_fft_metrics(target_patch)
                
                st.write("**Physics Check (FFT):**")
                c_fft1, c_fft2 = st.columns(2)
                c_fft1.image(target_patch, caption="Real Space", use_column_width=True, clamp=True)
                
                # Normalize FFT for display
                fft_disp = fft_mag - fft_mag.min()
                fft_disp = fft_disp / (fft_disp.max() + 1e-9)
                c_fft2.image(fft_disp, caption="Reciprocal Space", use_column_width=True, clamp=True)
                
                if crystal_score > 50:
                    st.success(f"💎 Crystalline (Score: {crystal_score:.1f})")
                else:
                    st.warning(f"☁️ Amorphous/Defect (Score: {crystal_score:.1f})")

                if st.button("Find More Like This 🔍"):
                    log_message(f"Training active learner on Region #{sel_rank}...")
                    new_score, new_map = train_classifier(res["features"], res["coords"], img.shape, target["id"])
                    st.session_state.current_score = new_score
                    st.session_state.current_map = new_map
                    st.session_state.mode = f"Supervised (Like #{sel_rank})"
                    st.session_state.history.append(new_map)
                    log_message("Model updated.")
                    st.rerun()

            with c_plot:
                embed = res["embedding"]
                if embed.shape[0] == len(score):
                    df = pd.DataFrame(embed, columns=["x", "y"])
                    df["score"] = score
                    df["size"] = 2
                    if target["id"] < len(df): df.loc[target["id"], "size"] = 10
                    fig = px.scatter(df, x="x", y="y", color="score", size="size", color_continuous_scale="Jet", title="Latent Space")
                    fig.update_layout(height=300, margin=dict(t=30, l=0, r=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)

    # --- TAB 3: Quantification ---
    with tab_quant:
        c_q, c_s = st.columns(2)
        with c_q:
            st.header("📏 Quantification")
            threshold = st.slider("Anomaly Threshold", 0.0, 1.0, 0.6)
            if st.session_state.current_map is not None:
                binary_map = st.session_state.current_map > threshold
                labels = measure.label(binary_map)
                n_features = labels.max()
                
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(img, cmap="gray", alpha=0.6)
                ax.imshow(binary_map, cmap="Reds", alpha=0.5)
                ax.axis("off")
                st.pyplot(fig)
                
                total_pixels = binary_map.sum()
                area_nm2 = total_pixels * (nm_per_pixel ** 2)
                st.metric("Count", int(n_features))
                st.metric("Total Area", f"{area_nm2:.1f} nm²")

        with c_s:
            st.header("💬 Semantic Search")
            st.info("Use CLIP to find defects by text description.")
            query = st.text_input("Query:", placeholder="e.g. 'linear cracks'")
            if st.button("Search"):
                if st.session_state.results:
                    log_message(f"Searching for: '{query}'")
                    res = st.session_state.results
                    with st.spinner("CLIP is searching..."):
                        text_scores, text_map = search_by_text(res["raw_patches"], query, img.shape, res["coords"])
                        st.session_state.current_score = text_scores
                        st.session_state.current_map = text_map
                        st.session_state.mode = f"Text: '{query}'"
                        st.session_state.history.append(text_map)
                        st.rerun()

    # --- TAB 4: Analytics ---
    with tab_analytics:
        st.header("📊 Efficiency Report")
        if st.session_state.current_map is not None:
            curr = st.session_state.current_map
            flat = np.sort(curr.flatten())[::-1]
            total_sig = flat.sum() + 1e-9
            x_vals = np.linspace(0, 100, 50)
            y_vals = []
            for t in x_vals:
                cutoff = int(len(flat) * (t/100))
                y_vals.append((flat[:cutoff].sum() / total_sig) * 100)
            
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(x_vals, y_vals, 'r-', linewidth=2, label="AI Scan")
            ax.plot(x_vals, x_vals, 'k--', alpha=0.3, label="Random")
            ax.set_ylabel("% Anomalies Found")
            ax.set_xlabel("% Time Spent")
            ax.legend()
            st.pyplot(fig)
            
            c1, c2 = st.columns(2)
            with c1:
                df_rep = pd.DataFrame({"Scan_Percentage": x_vals, "Signal_Captured": y_vals})
                st.download_button("Download CSV Report", df_rep.to_csv(index=False), "efficiency.csv", "text/csv")
            with c2:
                if st.session_state.results:
                    res = st.session_state.results
                    score = st.session_state.current_score
                    top_idx = np.argsort(score)[-10:][::-1]
                    top_regions = [{"rank": r+1, "x": int(res["coords"][i][1]), "y": int(res["coords"][i][0])} for r, i in enumerate(top_idx)]
                    
                    protocol = {"timestamp": "2025-12-18", "points": top_regions}
                    if st.download_button("Download Hardware Protocol (.json)", json.dumps(protocol, indent=2), "protocol.json", "application/json"):
                        st.balloons()
                        log_message("Protocol exported to controller.")

if __name__ == "__main__":
    main()
