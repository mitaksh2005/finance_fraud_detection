# Unified Risk Analysis & Fraud Detection System
### *Asymmetric Multi-Task TabNet + LightGBM Stacked Ensemble*

## 📌 Project Overview
This project implements a **Hybrid AI System** to detect fraudulent financial transactions while profiling user credit risk. It addresses the challenge of **"Task Interference"** in Multi-Task Learning by using a novel **Asymmetric Dual-Lane Architecture**.

The system achieves a **ROC-AUC of ~0.90+**, significantly outperforming traditional single-model baselines by combining Deep Learning (TabNet) with Gradient Boosting (LightGBM).

## 🚀 Key Features
* **Asymmetric Neural Architecture**: specialized "Private Lane" for behavioral fraud features (V-columns) to prevent them from corrupting the shared financial context.
* **Hybrid Stacked Ensemble**: Blends the global pattern recognition of **TabNet** with the precise decision boundaries of **LightGBM**.
* **Weighted Focal Loss**: Custom loss function to handle the extreme class imbalance (3.5% fraud rate).
* **Explainable AI**: SHAP (SHapley Additive exPlanations) dashboard to visualize the "Why" behind every rejection.

## 📂 Project Structure
```text
FINANCE-FRAUD-DETECTION/
├── data/                   # Raw and Processed Parquet files
├── notebooks/              # Research & Experimentation
│   ├── 01_data_ingestion.ipynb          # Memory optimization & Merging
│   ├── 02_feature_engineering.ipynb     # creating 'Magic Ratios'
│   ├── 03_eda_insights.ipynb            # Log-scale visualizations
│   ├── 04_tabnet_credit_baseline.ipynb  # Initial Single-Task experiments
│   ├── 05_ensemble_lightgbm.ipynb       # The Final Hybrid Ensemble Logic
│   ├── 06_evaluation_and_metrics.ipynb  # Performance Reports
│   └── 07_explainability_with_shap.ipynb# SHAP Analysis Plots
├── outputs/                # Saved Model Weights (.pth)
├── src/                    # Production Codebase
│   ├── data_loader.py      # Asymmetric Loader (Shared vs Private lanes)
│   ├── inference.py        # Hybrid Inference Script (TabNet + LGBM)
│   ├── loss_functions.py   # Weighted Focal Loss
│   ├── main.py             # Asymmetric Training Pipeline
│   ├── model.py            # MultiTaskTabNet Class
│   └── trainer.py          # PyTorch Training Loop
└── config.yaml             # Hyperparameters