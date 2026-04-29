# 🌤️ Weather Classification — AdaBoost

Weather type prediction using **Adaptive Boosting (AdaBoost)** on a dataset of 13,200 meteorological samples.  
The model classifies weather into four categories: **Cloudy, Rainy, Snowy, and Sunny**.

---

## 📋 Project Overview

This project builds an end-to-end machine learning pipeline for weather classification. It covers full exploratory data analysis, feature preprocessing, model training with AdaBoost, and comprehensive evaluation through multiple visualizations.

**Final Test Accuracy: 90.27%**

---

## 📁 Project Structure

```
weather-classification-adaboost/
│
├── weather_adaboost.ipynb          # Main notebook (EDA + Model + Evaluation)
├── weather_classification_data.csv # Dataset (13,200 samples)
├── .venv/                          # Virtual environment
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Total samples | 13,200 |
| Features | 12 |
| Target classes | 4 (Cloudy, Rainy, Snowy, Sunny) |
| Missing values | None |
| Class balance | Equal (3,300 per class) |

**Features used:**
- Temperature, Wind Speed, Precipitation (%), Atmospheric Pressure
- UV Index, Visibility (km), Cloud Cover
- Season, Location *(categorical)*
- Temp_Wind, Pressure_UV *(engineered interaction features)*

---

## 🔍 Notebook Contents

### 1. Exploratory Data Analysis (EDA)
- Weather type distribution (bar chart + pie chart)
- Seasonal distribution and stacked frequency by season
- Numerical feature histograms and box plots by weather type
- Violin plots for key features
- Correlation heatmap
- Average weather elements by location (heatmap)
- Temperature vs Wind Speed scatter plot
- Pairwise feature relationships (pairplot)

### 2. Preprocessing Pipeline
- Feature engineering: added `Temp_Wind` and `Pressure_UV` interaction features
- Median imputation + `StandardScaler` for numerical features
- Most-frequent imputation + `OneHotEncoder` for categorical features
- `ColumnTransformer` + `Pipeline` from scikit-learn
- 80/20 stratified train-test split

### 3. AdaBoost Model
- Base estimator: **Decision Stump** (`max_depth=3`)
- `n_estimators=200`, `learning_rate=0.5`
- Wrapped in a full scikit-learn `Pipeline`

### 4. Evaluation
- 5-fold cross-validation with per-fold bar chart
- Confusion matrix (counts + normalized)
- Misclassification analysis
- Feature importance (top 15)
- Learning curve
- Boosting progression curve (accuracy vs number of estimators)
- Performance radar chart
- Full classification report (precision, recall, F1 per class)

---

## ⚙️ Requirements

- Python 3.8+
- scikit-learn >= 1.0
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- jupyter

---

## 🚀 How to Run

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/weather-classification-adaboost.git
cd weather-classification-adaboost
```

### Step 2 — Create and activate a virtual environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install numpy pandas matplotlib seaborn plotly scikit-learn jupyter
```

### Step 4 — Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open `weather_adaboost.ipynb` from the Jupyter interface in your browser.

### Step 5 — Run the notebook

Click **Kernel → Restart & Run All** to execute all cells from top to bottom.

> ⚠️ Make sure `weather_classification_data.csv` is in the same folder as the notebook before running.

---

## 📈 Results

| Metric | Value |
|---|---|
| Test Accuracy | **90.27%** |
| Weighted Precision | ~0.90 |
| Weighted Recall | ~0.90 |
| Weighted F1-Score | ~0.90 |
| CV Mean Accuracy | ~0.91 |

---

## 🛠️ Tech Stack

`Python` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Plotly` · `Jupyter Notebook`
