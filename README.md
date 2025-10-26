# Credit Card Fraud Detection

A compact project to detect fraudulent credit card transactions using machine learning. This repository contains data exploration, preprocessing, modeling, and evaluation code to build and compare classifiers for fraud detection (highly imbalanced classification).

Table of contents
- [About](#about)
- [Dataset](#dataset)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Modeling & Evaluation](#modeling--evaluation)
- [Results](#results)
- [Reproducing experiments](#reproducing-experiments)
- [Contributing](#contributing)
- [License & Contact](#license--contact)

About
-----
This project demonstrates a typical credit card fraud detection pipeline:
- Exploratory data analysis (EDA)
- Data preprocessing and class imbalance handling
- Baseline and advanced classification models
- Model evaluation with metrics suitable for imbalanced data
- (Optional) Explainability using SHAP or similar tools

Dataset
-------
The repository uses the commonly used "Credit Card Fraud Detection" dataset (anonymized features, transactions made by European cardholders). If not included in the repo, download from Kaggle:
https://www.kaggle.com/datasets/ashrakattamer/credit-fraud-detection

Note: Make sure you follow the dataset license and Kaggle terms if you download and use the data.

Repository structure
--------------------
(Adjust these if your repo uses different names)
- data/                - raw and processed datasets (gitignore large/raw files)
- notebooks/           - EDA and modeling notebooks (Jupyter)
- src/                 - scripts and modules (data processing, training, evaluation)
- models/              - saved model artifacts
- reports/             - figures and evaluation reports
- requirements.txt     - Python package requirements
- README.md

Getting started
---------------
Prerequisites
- Python 3.8+ recommended
- Git

Create a virtual environment and install dependencies:
Unix / macOS:
```bash
git clone https://github.com/AshraqtTamer/Credit-card-fraud-detection.git
cd Credit-card-fraud-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
git clone https://github.com/AshraqtTamer/Credit-card-fraud-detection.git
cd Credit-card-fraud-detection
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you don't have a requirements.txt yet, these are common packages used in this project:
```text
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
jupyterlab
xgboost
lightgbm
shap
joblib
```

Usage
-----
- For exploratory analysis and interactive work, open the notebooks:
  ```bash
  jupyter lab
  # then open notebooks/your_notebook.ipynb
  ```

- If there are scripts in src/, example commands might be:
  ```bash
  python src/preprocess.py --input data/raw/creditcard.csv --output data/processed/
  python src/train.py --config configs/train_config.yaml
  python src/evaluate.py --model models/best_model.joblib --test data/processed/test.csv
  ```
Adjust script names and arguments to match your repo's files.

Modeling & Evaluation
---------------------
Because fraud detection is an imbalanced classification problem, focus on:
- Proper train/validation/test splitting (time-based split if data is temporal)
- Resampling or class-weight strategies (SMOTE, class_weight, focal loss)
- Evaluation metrics: Precision, Recall, F1-score, Precision-Recall AUC, ROC AUC (report Precision-Recall AUC as it is more informative for rare positive class)
- Calibration and confusion matrix analysis
- Explainability: SHAP or LIME for model interpretation

Suggested evaluation snippet (conceptual):
```python
from sklearn.metrics import classification_report, precision_recall_curve, auc

# predictions, probs, y_test assumed available
print(classification_report(y_test, preds))
precision, recall, _ = precision_recall_curve(y_test, probs)
pr_auc = auc(recall, precision)
print("PR AUC:", pr_auc)
```

Results
-------
Summarize the key model performance here (example):
- Best model: XGBoost with SMOTE + class weighting
- Precision: 0.82
- Recall: 0.76
- F1-score: 0.79
- PR AUC: 0.91

(Replace the above with your actual results and include plots under reports/.)

Reproducing experiments
-----------------------
- Keep experiments deterministic where possible (set random seeds)
- Log configurations and results (simple CSV logs, or use MLFlow/Weights & Biases)
- Save preprocessing pipeline and trained model (joblib or pickle) in models/

Contributing
------------
Contributions are welcome. Please:
1. Open an issue describing what you'd like to work on.
2. Create a branch with a descriptive name (e.g., feature/imbalance-handling).
3. Open a pull request with a clear description of changes.

License & Contact
-----------------
Specify your license (e.g., MIT). Add a LICENSE file at the repo root.

Author: AshraqtTamer
Contact: (add email or GitHub profile link)

Acknowledgements
----------------
- The Kaggle community dataset: "Credit Card Fraud Detection" (mlg-ulb)
- Standard ML/DS libraries and community resources
