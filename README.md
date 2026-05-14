# 🍷 Wine Type Classification — PCA, LDA & SVM from Scratch
 
> Classifying red vs white wine using dimensionality reduction and SVM, implemented both from scratch and with scikit-learn for comparison.
 
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![NumPy](https://img.shields.io/badge/NumPy-1.x-green)
 
---
 
##  Project Overview
 
This project tackles binary classification (red vs white wine) on the merged UCI Wine Quality dataset using three machine learning techniques — all implemented **from scratch** and then validated against scikit-learn equivalents:
 
- **PCA** — Principal Component Analysis for dimensionality reduction
- **LDA** — Linear Discriminant Analysis for supervised projection
- **SVM** — Support Vector Machine with RBF kernel for classification
---
 
##  Dataset
 
**Source:** `wine_quality_merged.csv` (merged red + white UCI Wine Quality datasets)
 
| Property | Value |
|----------|-------|
| Total samples | 6,497 |
| Features | 12 (11 physicochemical + type) |
| Target | `type` — red (0) or white (1) |
| Duplicates removed | Yes |
| Missing values | None |
 
**Features include:** fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free/total sulfur dioxide, density, pH, sulphates, alcohol.
 
---
 
##  Preprocessing Pipeline
 
1. **Duplicate removal** — 6,497 → cleaned dataset
2. **Label encoding** — `red → 0`, `white → 1`
3. **Stratified train/test split** — 80/20
4. **Standardization** — `StandardScaler` fitted on training set only
5. **Outlier clipping** — values beyond ±3σ (train statistics) clipped in both splits
---
 
## 🔬 Methods
 
###  PCA (from scratch)
Implemented using eigendecomposition of the covariance matrix:
1. Center data by subtracting training mean
2. Compute covariance matrix
3. Eigendecomposition → sort by descending eigenvalue
4. Project onto top-k principal components
Validated against `sklearn.decomposition.PCA` — projections match (up to sign flip, which is expected).
 
###  LDA (from scratch)
Binary LDA implemented via within-class scatter matrix:
1. Compute per-class means
2. Build within-class scatter matrices (S_W) and between-class scatter (S_B)
3. Solve for optimal projection vector W = S_W⁻¹(μ₁ − μ₂)
4. Classify using midpoint threshold between projected class means
Validated against `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`.
 
###  SVM with RBF Kernel (from scratch)
Custom kernel SVM using gradient descent:
- **Kernel:** RBF — K(x, xᵢ) = exp(−γ ‖x − xᵢ‖²)
- **Optimization:** Gradient descent on hinge loss with L2 regularization (C parameter)
- **γ:** auto-computed as 1 / (n_features × var(X)) if not provided
Validated against `sklearn.svm.SVC(kernel='rbf')`.
 
---
 
##  Results
 
| Model | Accuracy |
|-------|:--------:|
| LDA (from scratch) | ~99.4% |
| LDA (sklearn) | ~99.4% |
| SVM RBF (from scratch) | **99.44%** |
| SVM RBF (sklearn) | **99.53%** |
 
> The sklearn SVM edges out the scratch implementation by ~0.01% — the gap is due to sklearn's quadratic programming solver (libsvm) vs the gradient descent approximation used here.
 
---
 
##  Visualizations
 
- Histogram distributions for all numeric features
- Correlation heatmap
- Class distribution bar chart
- PCA scree plot (individual + cumulative explained variance)
- PCA 2D scatter — scratch vs sklearn side by side
- LDA 1D projection scatter — scratch vs sklearn
- Confusion matrices — scratch vs sklearn (side by side)
- Accuracy comparison bar charts
---
 
##  Getting Started
 
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```
 
Place `wine_quality_merged.csv` in the project root, then run the notebook.
 
---
 
##  Dependencies
 
```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
```
 
---
 
##  From Scratch vs Sklearn Summary
 
| Technique | Scratch Accuracy | Sklearn Accuracy | Gap |
|-----------|:---:|:---:|:---:|
| LDA | ~99.4% | ~99.4% | ≈ 0% |
| SVM (RBF) | 99.44% | 99.53% | ~0.01% |
 
Both scratch implementations reproduce sklearn results closely, confirming correctness of the math and implementation.
 
---
 
##  Authors
 
Built as a Machine Learning course project.
 
---
 
##  License
 
This project is licensed under the MIT License.
