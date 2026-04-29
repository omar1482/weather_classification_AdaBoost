import json

with open('weather_adaboost.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

tts_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'train_test_split(' in source:
            tts_idx = i
            break

fe_source = [
    "# Feature Engineering\n",
    "X['Temp_Wind'] = X['Temperature'] * X['Wind Speed']\n",
    "X['Pressure_UV'] = X['Atmospheric Pressure'] * X['UV Index']\n",
    "\n",
    "# Update feature columns lists since new columns were added\n",
    "categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()\n",
    "numeric_feature_cols = X.select_dtypes(include=np.number).columns.tolist()\n",
    "\n",
    "# Re-create preprocessor to pick up new numeric columns\n",
    "preprocessor = ColumnTransformer(transformers=[\n",
    "    ('num', numeric_transformer, numeric_feature_cols),\n",
    "    ('cat', categorical_transformer, categorical_cols),\n",
    "])\n"
]

fe_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": fe_source
}

nb['cells'].insert(tts_idx, fe_cell)

# Because we inserted a cell, all subsequent indices are shifted by 1.
# Re-iterate to find the adaboost cell and summary cell.
ada_idx = -1
sum_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'AdaBoostClassifier' in source and 'ada_pipeline.fit' in source:
            ada_idx = i
        if 'ADABOOST WEATHER CLASSIFICATION — EXECUTIVE SUMMARY' in source:
            sum_idx = i

tuned_ada_source = [
    "# Build AdaBoost with tuned configuration\n",
    "stump = DecisionTreeClassifier(max_depth=3, random_state=42)\n",
    "\n",
    "ada_clf = AdaBoostClassifier(\n",
    "    estimator=stump,\n",
    "    n_estimators=200,\n",
    "    learning_rate=0.5,\n",
    "    random_state=42\n",
    ")\n",
    "\n",
    "ada_pipeline = Pipeline(steps=[\n",
    "    ('preprocessor', preprocessor),\n",
    "    ('model', ada_clf),\n",
    "])\n",
    "\n",
    "print('Training AdaBoost...')\n",
    "ada_pipeline.fit(X_train, y_train)\n",
    "print('✅ Model trained successfully!')\n"
]

nb['cells'][ada_idx]['source'] = tuned_ada_source

# Update final summary cell
sum_source = nb['cells'][sum_idx]['source']
for i, line in enumerate(sum_source):
    if "Base estimator" in line:
        sum_source[i] = "print(f\"  Base estimator    : Decision Stump (max_depth=3)\")\n"
    elif "Learning rate" in line:
        sum_source[i] = "print(f\"  Learning rate     : 0.5\")\n"

nb['cells'][sum_idx]['source'] = sum_source

with open('weather_adaboost.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
