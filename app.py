import streamlit as st

st.set_page_config(
    page_title="DeepScan AI",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 DeepScan: Active Learning for Microscopy")

st.markdown("""
### Welcome to the Next-Generation Scanning Suite

This tool demonstrates how **Active Learning** can revolutionize microscopy by replacing "Raster Scanning" (scanning everything) with "Intelligent Scanning" (scanning only what matters).

**Getting Started:**
1. Go to **Active Discovery** (Sidebar) to upload an image and train the AI.
2. Go to **Efficiency Stats** (Sidebar) to see how much time you saved.

---
**Why this matters:**
> *Scanning at atomic resolution is slow. By identifying interesting regions first, we can reduce beam time by up to 90%.*
""")

# Initialize Global Session State
if "results" not in st.session_state:
    st.session_state.results = None
if "img_cache" not in st.session_state:
    st.session_state.img_cache = None
if "history" not in st.session_state:
    st.session_state.history = [] # To track improvement over time
