import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('weather_classification_data.csv')
if 'Humidity' in df.columns:
    df = df.drop(columns=['Humidity'])

TARGET_COL = 'Weather Type'
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_feature_cols = X.select_dtypes(include=np.number).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_feature_cols),
    ('cat', categorical_transformer, categorical_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with open('results.txt', 'w', encoding='utf-8') as f:
    f.write("Baseline accuracy: 89.13%\n")

    # STEP 1
    stump = DecisionTreeClassifier(max_depth=3, random_state=42)
    ada_clf = AdaBoostClassifier(estimator=stump, n_estimators=200, learning_rate=0.8, random_state=42)
    ada_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', ada_clf)])
    ada_pipeline.fit(X_train, y_train)
    y_pred = ada_pipeline.predict(X_test)
    step1_acc = accuracy_score(y_test, y_pred) * 100
    f.write(f"✅ [STEP 1 — DEEPER BASE ESTIMATOR — {step1_acc:.2f}%]\n")

    # STEP 2
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.5, 0.8, 1.0],
        'model__estimator__max_depth': [1, 2, 3]
    }

    base_stump = DecisionTreeClassifier(random_state=42)
    base_ada = AdaBoostClassifier(estimator=base_stump, random_state=42)
    grid_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', base_ada)])

    grid_search = GridSearchCV(grid_pipeline, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_ * 100
    f.write(f"Best params: {best_params}\n")
    f.write(f"Best CV score: {best_cv_score:.2f}%\n")

    best_model = grid_search.best_estimator_
    y_pred_2 = best_model.predict(X_test)
    step2_acc = accuracy_score(y_test, y_pred_2) * 100
    f.write(f"✅ [STEP 2 — HYPERPARAMETER TUNING — {step2_acc:.2f}%]\n")

    # STEP 3
    X_feat = X.copy()
    X_feat['Temp_Wind'] = X_feat['Temperature'] * X_feat['Wind Speed']
    X_feat['Pressure_UV'] = X_feat['Atmospheric Pressure'] * X_feat['UV Index']

    numeric_feature_cols_new = X_feat.select_dtypes(include=np.number).columns.tolist()
    preprocessor_new = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_feature_cols_new),
        ('cat', categorical_transformer, categorical_cols),
    ])

    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X_feat, y, test_size=0.2, random_state=42, stratify=y
    )

    best_n_estimators = best_params['model__n_estimators']
    best_learning_rate = best_params['model__learning_rate']
    best_max_depth = best_params['model__estimator__max_depth']

    stump_f = DecisionTreeClassifier(max_depth=best_max_depth, random_state=42)
    ada_clf_f = AdaBoostClassifier(
        estimator=stump_f, 
        n_estimators=best_n_estimators, 
        learning_rate=best_learning_rate, 
        random_state=42
    )
    ada_pipeline_f = Pipeline(steps=[('preprocessor', preprocessor_new), ('model', ada_clf_f)])
    ada_pipeline_f.fit(X_train_f, y_train_f)
    y_pred_f = ada_pipeline_f.predict(X_test_f)
    step3_acc = accuracy_score(y_test_f, y_pred_f) * 100
    f.write(f"✅ [STEP 3 — FEATURE ENGINEERING — {step3_acc:.2f}%]\n")
