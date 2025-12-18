import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("📊 Efficiency Benchmarks")

if "history" not in st.session_state or not st.session_state.history:
    st.warning("No data yet! Go to 'Active Discovery' and run an analysis first.")
    st.stop()

st.markdown("""
This dashboard simulates the **Microscope Beam Time** saved by using your AI model. 
Instead of scanning 100% of the pixels, we only scan the high-probability regions.
""")

# --- Simulation Logic ---
def calculate_captured_signal(scan_map, threshold_percent):
    """
    Simulates: If we scan top X% of pixels, how much total 'signal' (intensity) do we capture?
    """
    flat_map = scan_map.flatten()
    # Sort pixels by AI score
    sorted_pixels = np.sort(flat_map)[::-1]
    total_signal = sorted_pixels.sum()
    
    # Calculate cumulative sum
    cum_signal = np.cumsum(sorted_pixels)
    
    # Find index for X% of pixels
    n_pixels = len(flat_map)
    idx = int(n_pixels * (threshold_percent / 100))
    
    captured = cum_signal[idx]
    return (captured / total_signal) * 100

# --- Metrics ---
latest_map = st.session_state.history[-1]["map"]
baseline_map = st.session_state.history[0]["map"]

col1, col2, col3 = st.columns(3)

# Metric 1: Signal capture at 10% Scan
capture_10_ai = calculate_captured_signal(latest_map, 10)
capture_10_base = calculate_captured_signal(baseline_map, 10) # Roughly random if uninformative
capture_10_random = 10.0 # Random scanning gets 10% signal in 10% time

col1.metric("Signal Captured (10% Scan)", f"{capture_10_ai:.1f}%", delta=f"{capture_10_ai - capture_10_random:.1f}% vs Random")

# Metric 2: Time to capture 80% Signal
def time_to_capture(scan_map, target_signal=0.8):
    flat = np.sort(scan_map.flatten())[::-1]
    cum = np.cumsum(flat)
    total = cum[-1]
    # Find first index where cum > target * total
    idx = np.searchsorted(cum, target_signal * total)
    return (idx / len(flat)) * 100

time_ai = time_to_capture(latest_map)
time_saved = 100 - time_ai

col2.metric("Scan Time Needed (80% Quality)", f"{time_ai:.1f}%", delta=f"-{time_saved:.1f}% Saved", delta_color="normal")
col3.metric("Improvement Steps", len(st.session_state.history))

# --- Chart ---
st.subheader("Scan Efficiency Curve (ROC-like)")

x_vals = np.linspace(0, 100, 100)
y_ai = [calculate_captured_signal(latest_map, x) for x in x_vals]
y_random = x_vals # Linear line

fig, ax = plt.subplots()
ax.plot(x_vals, y_ai, label="Active Learning (AI)", color="red", linewidth=2)
ax.plot(x_vals, y_random, label="Raster Scan (Random)", linestyle="--", color="gray")

ax.set_xlabel("% of Image Scanned (Time)")
ax.set_ylabel("% of Interesting Features Found")
ax.set_title("Performance: AI vs Standard Scan")
ax.legend()
ax.grid(True, alpha=0.3)
ax.fill_between(x_vals, y_ai, y_random, color='red', alpha=0.1)

st.pyplot(fig)

st.markdown("""
**How to read this chart:**
* The **Red Line** is your model. It shoots up quickly, meaning it finds the important stuff first.
* The **Gray Line** is a standard microscope scan.
* The **Area between lines** represents the *Efficiency Gain*.
""")
