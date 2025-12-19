#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py: Configuration settings for the DeepScan Pro application.
Stores constants, model names, default parameters, and hardware simulation settings.
"""

# Page configuration
PAGE_CONFIG = {
    "page_title": "DeepScan Pro",
    "page_icon": "🔬",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Available Neural Network Backbones
AVAILABLE_MODELS = [
    "regnet_y_400mf", 
    "convnext_tiny", 
    "resnet50"
]

# Default Analysis Parameters
DEFAULT_PATCH_SIZES = [32, 64]
DEFAULT_STRIDE_DIVISOR = 2
DEFAULT_PCA_DIM = 50

# Hardware Simulation Constants
HARDWARE_NAME = "STEM-v3 Controller"
CONNECTION_DELAY = 0.8
