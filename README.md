
# Active Learning for High-Entropy Alloy Design

This repository implements various active learning (AL), multi-fidelity active learning (MFAL), and regression-based surrogate modeling techniques to accelerate the discovery of high-performance **High Entropy Alloys (HEAs)**. The dataset and primary objective align with the work of:

> **Rao et al., 2022** – _Machine learning–enabled high-entropy alloy discovery_, Science, 378(6615), 78-85.  
> [DOI: 10.1126/science.abo5810](https://www.science.org/doi/10.1126/science.abo5810)

## 📁 Repository Structure

| File/Folder     | Description |
|----------------|-------------|
| `Data_base.csv` | Source dataset extracted from Rao et al. (2022) containing composition and thermal expansion coefficients (TEC) of HEAs. |
| `AL_GPR.py`     | Active learning using **Gaussian Process Regression** (GPR) with acquisition functions like EI and LCB. |
| `AL_RFV.py`     | Active learning using **Random Forest Regression** with exploration vs. exploitation strategies. |
| `MFAL.py`       | **Multi-Fidelity Active Learning (MFAL)** framework incorporating both low- and high-fidelity surrogate models with dynamic fidelity-aware acquisition strategies. |
| `ANN.py`        | Neural Network (ANN)-based regression model with PCA preprocessing and outlier filtering. |
| `GPR.py`        | Standalone Gaussian Process Regression model with cross-validation and model diagnostics. |
| `ML.py`         | Utility or orchestration script for experiments or unified execution (assumed based on filename). |
| `preprocessing.ipynb` | Jupyter notebook for preprocessing, visualization, and initial exploration. |
| `Results/`      | Pickle files and generated plots storing experiment outputs and analysis results. |

## 🧪 Key Features

- **Active Learning Loops** using Expected Improvement (EI) and Lower Confidence Bound (LCB).
- **Multi-fidelity selection** balancing low/high cost evaluations dynamically.
- **Dimensionality reduction** via PCA.
- **Outlier detection** using Isolation Forest for cleaner model training.
- Evaluation with:
  - Simple & cumulative regret
  - Convergence thresholds (within 10%, 5%, 1% of true minimum)
  - Spatial diversity metrics
- Visualization of exploration vs. exploitation and model performance.

## 📊 Results

All major results (plots, pickle files, model diagnostics) are stored in the `Results/` folder. These include:
- Optimization convergence plots
- Regret analysis
- Predicted vs actual TEC comparisons
- Run-wise statistics and top-performing compositions

![AL_anim](https://github.com/user-attachments/assets/22ca607f-df88-4f9e-a917-d2051c5d6d1f)

## 🛠 Dependencies

Install via pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch gpytorch botorch tqdm



https://github.com/user-attachments/assets/ee4dcd8f-2219-4520-ad10-58f7ac539639

