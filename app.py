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
from utils import load_image_grayscale, get_default_image, normalize
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

    # --- MAIN TITLE ---
    st.title("🔬 DeepScan Pro")
    st.caption("🚀 Intelligent Active Learning for Atomic Microscopy | Purdue University")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ System Control")
        
        # Hardware Status
        st.subheader("🔌 Hardware Link")
        if not st.session_state.connected:
            st.error("🔴 Offline")
            if st.button("⚡ Connect Microscope"):
                with st.spinner("Handshaking with Controller..."):
                    time.sleep(0.8)
                    log_message(f"Connected to {HARDWARE_NAME}")
                st.session_state.connected = True
                st.rerun()
        else:
            st.success(f"🟢 Online ({HARDWARE_NAME})")
            if st.button("❌ Disconnect"):
                log_message("Disconnected from hardware.")
                st.session_state.connected = False
                st.rerun()

        st.divider()

        # System Logs
        st.subheader("📟 Live Terminal")
        log_text = "\n".join(st.session_state.logs)
        st.text_area("Console Output", value=log_text, height=120, disabled=True)
        
        st.divider()

        # Input Data
        st.subheader("💾 Input Data")
        uploaded = st.file_uploader("Upload Scan", type=["png", "jpg", "tif"])
        
        if uploaded:
            img_bytes = uploaded.getvalue()
            img = load_image_grayscale(io.BytesIO(img_bytes))
            st.success("✅ File Loaded")
        elif st.session_state.img_cache is not None:
            img = st.session_state.img_cache 
        else:
            img = get_default_image()
            st.session_state.img_cache = img
            st.info("ℹ️ Demo Mode")

        if st.session_state.img_cache is None or not np.array_equal(img, st.session_state.img_cache):
            st.session_state.img_cache = img
            st.session_state.results = None
            st.session_state.history = []
            log_message("New buffer loaded.")

        # Configuration
        st.subheader("🧠 Model Config")
        backbone = st.selectbox("Vision Backbone", AVAILABLE_MODELS, index=1)
        nm_per_pixel = st.number_input("Scale (nm/px)", value=1.0, min_value=0.01, format="%.2f")

        with st.expander("🛠️ Advanced Settings"):
            patch_size_input = st.multiselect("Patch Sizes", [32, 64, 128], default=DEFAULT_PATCH_SIZES)
            stride_val = st.slider("Overlap", 1, 4, 2)
            pca_dim = st.slider("PCA Dim", 10, 100, DEFAULT_PCA_DIM)

        st.divider()
        if st.button("🔥 Reset System"):
            st.session_state.clear()
            st.rerun()

    # --- TABS ---
    tab_about, tab_run, tab_quant, tab_analytics = st.tabs(["💡 Logic", "👁️ Visual Scan", "📐 Metrology", "📈 Analytics"])

    # --- TAB 1: LOGIC (UPDATED AUTHOR FORMAT) ---
    with tab_about:
        st.header("How DeepScan Works")
        
        # Author Card (Updated Format)
        with st.container(border=True):
            st.markdown("""
            **Md Habibur Rahman** School of Materials Engineering, Purdue University, West Lafayette, IN 47907, USA  
            rahma103@purdue.edu
            """)

        if os.path.exists("workflow.png"):
            st.image("workflow.png", caption="Figure 1: The Active Learning Architecture.", width=900)
        else:
            st.warning("⚠️ Diagram missing.")

        st.info("DeepScan transforms passive images into active hardware protocols.", icon="⚡")
        st.markdown("---")

        # --- THE "NICE MODE" ACCORDIONS ---
        with st.expander("👁️ **Step 1: The 'See' Phase (Neural Extraction)**", expanded=True):
            st.markdown("""
            The system decomposes the scan into overlapping patches. 
            A **RegNet** or **ConvNeXt** extracts semantic features $z$ from each patch $x$.
            """)
            st.latex(r"z = f_{\theta}(x) \in \mathbb{R}^{1024}")

        with st.expander("🧠 **Step 2: The 'Think' Phase (Anomaly Detection)**", expanded=False):
            st.markdown("""
            We use **Isolation Forests** to build a 'Surprisal Map' of the sample.
            Rare features (defects) are isolated faster than common lattice structures.
            """)
            st.latex(r"s(x, n) = 2^{- \frac{E(h(x))}{c(n)}}")

        with st.expander("🎓 **Step 3: The 'Learn' Phase (Teacher Mode)**", expanded=False):
            st.markdown("""
            When you click a target, we train a **Few-Shot Classifier** instantly.
            This updates the probability map to find specific features like vacancies or cracks.
            """)
            st.latex(r"P(y=1|z) = \frac{1}{1 + e^{-(w^T z + b)}}")

    # --- TAB 2: VISUAL SCAN (SMALLER BUTTON) ---
    with tab_run:
        st.header("🚀 AI-Powered Discovery")
        
        # Button is now smaller (removed use_container_width=True)
        if st.button("✨ Initialize DeepScan", type="primary"):
            log_message(f"Initializing {backbone}...")
            with st.spinner(f"Processing with {backbone}..."):
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
            c_sim, c_view = st.columns([1, 2])
            with c_sim:
                st.success("Analysis Ready")
                st.markdown("### 🎮 Stage Control")
                if st.button("▶️ Simulate Live Scan", type="secondary"):
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
                            ax.plot(px_coords, py, 'r--', linewidth=2, alpha=0.8, label="Path")
                        
                        last_r = current_regions[-1]
                        rect = mpatches.Rectangle((last_r["j"], last_r["i"]), last_r["size"], last_r["size"], linewidth=3, edgecolor="#00FF7F", facecolor="none")
                        ax.add_patch(rect)
                        ax.text(last_r["j"], max(0, last_r["i"]-5), str(last_r["rank"]), color="#00FF7F", fontsize=14, weight="bold")
                        
                        ax.axis("off")
                        ax.set_title(f"Scanning Region #{last_r['rank']}...", color="#00FF7F")
                        
                        placeholder.pyplot(fig)
                        plt.close(fig)
                        time.sleep(0.4)
                    
                    log_message("Scan simulation completed.")
                    st.rerun()

            # --- Static View ---
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔭 Microscope View")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(img, cmap="gray")
                
                path_y = [r["i"] + r["size"]//2 for r in top_regions]
                path_x = [r["j"] + r["size"]//2 for r in top_regions]
                
                ax.plot(path_x, path_y, 'r--', linewidth=1.5, alpha=0.8, label="Optimized Path")
                ax.scatter(path_x[0], path_y[0], c='#00FF7F', s=100, zorder=5, label="Start Point")
                
                for r in top_regions:
                    rect = mpatches.Rectangle((r["j"], r["i"]), r["size"], r["size"], linewidth=2, edgecolor="#00FF7F", facecolor="none")
                    ax.add_patch(rect)
                    ax.text(r["j"], max(0, r["i"]-5), str(r["rank"]), color="#00FF7F", fontsize=12, weight="bold")
                
                ax.legend(loc="lower right")
                ax.axis("off")
                st.pyplot(fig)

            with col2:
                st.markdown(f"#### 🗺️ Priority Map ({st.session_state.mode})")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(img, cmap="gray", alpha=0.4)
                im = ax.imshow(st.session_state.current_map, cmap="jet", alpha=0.6)
                ax.axis("off")
                plt.colorbar(im, ax=ax)
                st.pyplot(fig)

            st.divider()
            
            # Gallery
            st.markdown("#### 🔍 Detected Anomalies")
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
                st.info("Select a region to train the AI.")
                sel_rank = st.selectbox("Select Target:", [r["rank"] for r in top_regions])
                target = next(r for r in top_regions if r["rank"] == sel_rank)
                
                # FFT Check
                target_patch = img[target['i']:target['i']+target['size'], target['j']:target['j']+target['size']]
                fft_mag, crystal_score = compute_fft_metrics(target_patch)
                
                st.markdown("**⚛️ Physics Check (FFT)**")
                c_fft1, c_fft2 = st.columns(2)
                c_fft1.image(target_patch, caption="Real Space", use_column_width=True, clamp=True)
                
                fft_disp = fft_mag - fft_mag.min()
                fft_disp = fft_disp / (fft_disp.max() + 1e-9)
                c_fft2.image(fft_disp, caption="Reciprocal", use_column_width=True, clamp=True)
                
                if crystal_score > 50:
                    st.success(f"💎 Crystalline ({crystal_score:.0f})")
                else:
                    st.warning(f"☁️ Amorphous ({crystal_score:.0f})")

                if st.button("🔎 Find More Like This"):
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
                    fig = px.scatter(df, x="x", y="y", color="score", size="size", color_continuous_scale="Jet", title="Latent Feature Space")
                    fig.update_layout(height=350, margin=dict(t=30, l=0, r=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)

    # --- TAB 3: METROLOGY ---
    with tab_quant:
        c_q, c_s = st.columns(2)
        with c_q:
            st.header("📐 Quantification")
            threshold = st.slider("Segmentation Threshold", 0.0, 1.0, 0.6)
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
                st.metric("Feature Count", int(n_features))
                st.metric("Total Area", f"{area_nm2:.1f} nm²")

        with c_s:
            st.header("💬 Semantic Search")
            st.info("Type a description to find specific defects.")
            query = st.text_input("Query:", placeholder="e.g. 'linear cracks', 'hexagonal lattice'")
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

    # --- TAB 4: ANALYTICS ---
    with tab_analytics:
        st.header("📈 Efficiency Report")
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
            ax.plot(x_vals, y_vals, 'r-', linewidth=2, label="DeepScan (AI)")
            ax.plot(x_vals, x_vals, 'k--', alpha=0.3, label="Random Scan")
            ax.set_ylabel("% Signal Captured")
            ax.set_xlabel("% Time Spent")
            ax.legend()
            st.pyplot(fig)
            
            c1, c2 = st.columns(2)
            with c1:
                df_rep = pd.DataFrame({"Scan_Percentage": x_vals, "Signal_Captured": y_vals})
                st.download_button("💾 Download Report (CSV)", df_rep.to_csv(index=False), "efficiency.csv", "text/csv")
            with c2:
                if st.session_state.results:
                    res = st.session_state.results
                    score = st.session_state.current_score
                    top_idx = np.argsort(score)[-10:][::-1]
                    top_regions = [{"rank": r+1, "x": int(res["coords"][i][1]), "y": int(res["coords"][i][0])} for r, i in enumerate(top_idx)]
                    
                    protocol = {"timestamp": "2025-12-18", "author": "rahma103", "points": top_regions}
                    if st.download_button("🤖 Export Protocol (.json)", json.dumps(protocol, indent=2), "protocol.json", "application/json"):
                        st.balloons()
                        log_message("Protocol sent to microscope controller.")

if __name__ == "__main__":
    main()
