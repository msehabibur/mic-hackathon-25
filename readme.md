# 🔬 DeepScan Pro: Intelligent Microscopy Agent

> **Winner of [Hackathon Name] (Optional Placeholder)**
> An Active Learning pipeline that turns passive microscopes into intelligent discovery agents.

## 🚀 The Problem
Scanning samples at atomic resolution (STEM/SEM) is slow and expensive. Researchers waste 90% of beam time scanning empty background to find rare defects.

## 💡 The Solution
**DeepScan Pro** connects to the microscope and uses **Unsupervised AI** to "look" before it scans.
1.  **See:** Decomposes low-res preview images into neural patches (RegNet/ConvNeXt).
2.  **Think:** Identifies anomalies using Isolation Forests & PCA.
3.  **Act:** Generates a machine-readable protocol to drive the hardware *only* to interesting regions.

## ✨ Key Features
* **Zero-Shot Detection:** Finds defects (cracks, particles, vacancies) without ANY training data.
* **Teacher Mode:** "Human-in-the-loop" functionality. Click one defect, and the AI finds all others instantly.
* **Natural Language Search:** Type "linear cracks" and our integrated CLIP model finds them.
* **Protocol Export:** Outputs `.json` coordinates for microscope stage controllers.

## 🛠️ Tech Stack
* **Core:** Python, PyTorch, Scikit-Learn
* **Vision:** RegNet, ConvNeXt, CLIP (HuggingFace)
* **UI:** Streamlit, Plotly, Matplotlib
* **Math:** UMAP, PCA, Isolation Forests

## ⚡ Quick Start
1.  Install dependencies: `pip install -r requirements.txt`
2.  Run the app: `streamlit run app.py`
3.  Upload a STEM/SEM image (or use the built-in demo).
