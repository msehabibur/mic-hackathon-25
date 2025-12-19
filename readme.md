# 🔬 DeepScan Pro: Intelligent Microscopy Agent

<img width="2816" height="1536" alt="image" src="https://github.com/user-attachments/assets/93825ade-7fcc-4a4d-9d67-2ad8f4c1b751" />


[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deepscanpro.streamlit.app/)

> **An Active Learning pipeline that turns passive microscopes into intelligent discovery agents.**

## 🔗 Try the Live Demo
👉 **[Click here to run DeepScan Pro in your browser](https://deepscanpro.streamlit.app/)**

---

## 🚀 The Problem
Scanning samples at atomic resolution (STEM/SEM) is slow and expensive. Researchers waste **90% of beam time** scanning empty background just to find a few rare defects.

## 💡 The Solution
**DeepScan Pro** connects to the microscope and uses **Unsupervised AI** to "look" before it scans.

1.  **See:** Decomposes low-res preview images into neural patches (RegNet/ConvNeXt).
2.  **Think:** Identifies anomalies using Isolation Forests & PCA.
3.  **Act:** Generates a machine-readable protocol to drive the hardware *only* to interesting regions.

## ✨ Key Features
* **Zero-Shot Detection:** Finds defects (cracks, particles, vacancies) without ANY training data.
* **Teacher Mode:** "Human-in-the-loop" functionality. Click one defect, and the AI finds all others instantly.
* **Natural Language Search:** Type *"linear cracks"* and our integrated CLIP model finds them.
* **Protocol Export:** Outputs `.json` coordinates for microscope stage controllers.

## 🛠️ Tech Stack
* **Core:** Python, PyTorch, Scikit-Learn
* **Vision:** RegNet, ConvNeXt, CLIP (HuggingFace)
* **UI:** Streamlit, Plotly, Matplotlib
* **Math:** UMAP, PCA, Isolation Forests

## ⚡ Quick Start

### Option 1: Live Cloud App
No installation required. Just visit **[deepscanpro.streamlit.app](https://deepscanpro.streamlit.app/)**.

### Option 2: Local Installation
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the app:
    ```bash
    streamlit run app.py
    ```
4.  Upload a STEM/SEM image (or use the built-in demo).
