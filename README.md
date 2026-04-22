# Heart Disease Prediction — Production Project

## Project Structure

```
heart_disease_project/
├── eda.py            # Exploratory Data Analysis
├── preprocess.py     # Data preprocessing pipeline (KNN Imputer + Scaler + SMOTE)
├── train.py          # Model training & evaluation for ALL 14 models
├── app.py            # Streamlit web interface
├── requirements.txt
├── heart.csv         # ← Place your dataset here
├── models/           # Auto-created: saved models + preprocessor
└── figures/          # Auto-created: all EDA and evaluation plots
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place the dataset
Copy your CSV file and rename/point it correctly:
```
C:\Users\aryan\Desktop\heart.csv  →  heart_disease_project\heart.csv
```
Or edit the path in `train.py` and `eda.py` at the bottom (`if __name__ == "__main__":`).

---

## Run Order

### Step 1 — EDA (optional but recommended)
```bash
python eda.py
```
Generates charts in `figures/` and prints dataset insights.

### Step 2 — Train all models
```bash
python train.py
```
- Preprocesses data (KNN Imputer + StandardScaler + SMOTE)
- Trains all 14 models (no additions, no removals)
- Saves models to `models/`
- Saves evaluation metrics to `models/results.csv`

### Step 3 — Launch the web app
```bash
streamlit run app.py
```
Opens at: http://localhost:8501

---

## Models Included (from original notebook)

| # | Model               | Category  |
|---|---------------------|-----------|
| 1 | Logistic Regression | Base      |
| 2 | SVM                 | Base      |
| 3 | KNN                 | Base      |
| 4 | Decision Tree       | Base      |
| 5 | Naive Bayes         | Base      |
| 6 | Random Forest       | Advanced  |
| 7 | Extra Trees         | Advanced  |
| 8 | XGBoost             | Advanced  |
| 9 | LightGBM            | Advanced  |
|10 | AdaBoost            | Ensemble  |
|11 | Bagging             | Ensemble  |
|12 | Voting Ensemble     | Ensemble  |
|13 | Stacking Ensemble   | Ensemble  |
|14 | MLP (DL)            | Neural    |

---

## Example Input Values

These are typical values for a patient **WITH** heart disease:

| Feature   | Value | Description                          |
|-----------|-------|--------------------------------------|
| age       | 63    | Age in years                         |
| sex       | 1     | 1 = Male                             |
| cp        | 3     | Chest pain type (3 = asymptomatic)   |
| trestbps  | 145   | Resting blood pressure (mm Hg)       |
| chol      | 233   | Serum cholesterol (mg/dl)            |
| fbs       | 1     | Fasting blood sugar > 120 mg/dl      |
| restecg   | 0     | Resting ECG results                  |
| thalach   | 150   | Max heart rate achieved              |
| exang     | 0     | Exercise induced angina              |
| oldpeak   | 2.3   | ST depression induced by exercise    |
| slope     | 0     | Slope of peak exercise ST segment    |
| ca        | 0     | Number of major vessels (0–4)        |
| thal      | 1     | 1=Normal, 2=Fixed defect, 3=Rev.     |

---

## Dataset Column Details

| Column   | Type        | Range / Values        |
|----------|-------------|-----------------------|
| age      | Integer     | 29 – 77               |
| sex      | Binary      | 0 = Female, 1 = Male  |
| cp       | Integer     | 0–3                   |
| trestbps | Integer     | 94–200                |
| chol     | Integer     | 126–564               |
| fbs      | Binary      | 0 or 1                |
| restecg  | Integer     | 0–2                   |
| thalach  | Integer     | 71–202                |
| exang    | Binary      | 0 or 1                |
| oldpeak  | Float       | 0.0–6.2               |
| slope    | Integer     | 0–2                   |
| ca       | Integer     | 0–4                   |
| thal     | Integer     | 0–3                   |
| target   | Binary      | 0 = No Disease, 1 = Disease |

---

## Preprocessing Pipeline

Mirrors the original notebook exactly:
1. **Numerical cols** → `KNNImputer(n_neighbors=5)` + `StandardScaler`
2. **Binary col (sex)** → `SimpleImputer(strategy='most_frequent')`
3. **SMOTE** applied on training data only (not test set)
4. 80/20 stratified train-test split

---

## Notes
- No model was added, removed, or had hyperparameters changed
- The UI auto-generates input fields from the dataset columns
- All models are loaded (not retrained) in the web app
