# AI-Captcha-Solver

An AI-powered CAPTCHA solver built with PyTorch that achieves up to **100%** accuracy on the Fournier CAPTCHA Version 2 dataset. This repository provides a high-performance training pipeline, mixed-precision GPU acceleration, and an easy-to-use inference script.

---

## 🔍 Features

- **ResNet-18 Backbone** with grayscale input  
- **Global Average Pooling + Dropout** head  
- **Mixed-Precision Training (AMP)** for faster GPU throughput  
- **OneCycleLR** scheduler for rapid convergence  
- **Early Stopping & Checkpointing** on best validation accuracy  
- **Data Augmentations**: random affine, color jitter, random erasing  
- **Windows-safe Multiprocessing** (top-level Dataset class + `__main__` guard)  
- **Graceful Ctrl+C Handling** to save the best model on interruption  

---

## 📦 Requirements

- Python 3.12  
- PyTorch (CUDA-enabled)  
- torchvision  
- scikit-learn  
- Pillow  
- kagglehub  

---

## ⚙️ Installation

1. **Clone this repo**  
   ```bash
   git clone https://github.com/your-username/AI-Captcha-Solver.git
   cd AI-Captcha-Solver

---
2. **Create & activate a Python 3.12 venv**

    ```bash
    Copy
    Edit
    python3.12 -m venv venv
    source venv/bin/activate    # Linux / macOS
    venv\Scripts\activate       # Windows
---      
3. **Install dependencies**
      ```bash
      pip install -r requirements.txt

---     
4. **Run Script**
      ```bash
      python train.py
   
