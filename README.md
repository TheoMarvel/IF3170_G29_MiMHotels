# Tugas Besar 2 IF3170 Inteligensi Artifisial — Group 29 (MiMHotels)

Binary classification project predicting hotel booking cancellations (`is_canceled`) for **IF3170 Inteligensi Artifisial (2026/2027)**. The project implements five machine learning algorithms from scratch using NumPy—**K-Nearest Neighbors (KNN)**, **Decision Tree Learning (C4.5)**, **Logistic Regression**, **Support Vector Machine (SVM)**, and **Naive Bayes**—alongside a modular preprocessing pipeline, cost-sensitive and standard evaluation metrics, side-by-side comparison against `scikit-learn`, model serialization (`.pkl`), hyperparameter tuning, and Kaggle submission generation.

## Repository Structure

```text
IF3170_G29_MiMHotels/
├── dataset/                                  # Raw train, test, and sample submission CSVs
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── docs/                                     # Documentation assets
├── output/
│   ├── Notebook_G29_MiMHotels.ipynb
│   ├── models/                               # Saved .pkl models and fitted pipeline
│   └── submission*.csv                       # Generated Kaggle submission files
└── src/
    ├── base/                                 # Base classifier & pickle serializer
    ├── dependencies/                         # Split criteria, distance metrics, kernels, optimizers, regularizers, and decision tree node
    ├── preprocessing/                        # Cleaning, encoding, scaling, imbalance handling, feature selection, PCA
    ├── models/
    │   ├── scratch/                          # From-scratch NumPy implementations (KNN, DTL C4.5, LogReg, SVM, Naive Bayes)
    │   └── sklearn_model/                    # Scikit-learn wrapper counterparts for benchmarking
    └── evaluation/                           # Train/val splitting, metrics, cost-sensitive evaluator, comparison utilities
```

---

## Setup & How to Run

### Prerequisites

- **Python**: 3.10+ (developed and tested on Python 3.11)
- **Libraries**: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `jupyter` / `ipykernel`

---

### Option 1: Run Locally (Jupyter / VS Code)

1. **Clone the repository** and enter the project directory:
   ```bash
   git clone https://github.com/TheoMarvel/IF3170_G29_MiMHotels.git
   cd IF3170_G29_MiMHotels
   ```

2. **Create and activate a Python environment**, then install the required packages:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # On Windows: .venv\Scripts\activate
   pip install numpy pandas matplotlib scikit-learn jupyter ipykernel
   ```

3. **Verify the dataset files** are located inside `dataset/`:
   - `dataset/train.csv`
   - `dataset/test.csv`
   - `dataset/sample_submission.csv`

4. **Open and run the notebook**:
   - Launch `jupyter notebook` or open `output/Notebook_Tugas_Besar_2_IF3170_Inteligensi_Artifisial_2026_2027.ipynb` directly in VS Code.
   - Select your virtual/conda Python kernel.
   - Run all cells sequentially from top to bottom (**Run All**).
   - *(Note: The library import cell automatically resolves the repository root so `from src import *` works even when the notebook runs inside `output/`.)*

---

### Option 2: Run on Google Colab

1. Open `output/Notebook_Tugas_Besar_2_IF3170_Inteligensi_Artifisial_2026_2027.ipynb` in Google Colab.
2. Run the Colab setup cell at the top of the notebook to clone IF3170_G29_MiMHotels into /content/IF3170_G29_MiMHotels and add it to sys.path.
3. Run the remaining cells sequentially from top to bottom.

---

### Expected Outputs

Running the notebook end-to-end produces:
- Serialized from-scratch models and the fitted preprocessing pipeline in `output/models/` (`*.pkl`)
- Evaluation tables, training/validation comparison plots, and hyperparameter tuning charts inline in the notebook
- Kaggle prediction CSVs in `output/` (such as `output/submission1.csv`)

---

## Job Division

| NIM | Name | Responsibilities / Tasks |
| :--- | :--- | :--- |
| `10623061` | Theodore Marvel Suranton | End-to-end project implementation: data understanding, preprocessing pipeline, all from-scratch models (KNN, DTL C4.5, LogReg, SVM, Naive Bayes), `scikit-learn` comparison, hyperparameter tuning, model serialization, error analysis, and Kaggle submission. |
