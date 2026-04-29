import json

with open('weather_adaboost.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'ada_clf = AdaBoostClassifier(' in source:
            new_source = """# Build AdaBoost with decision stump as base estimator
stump = DecisionTreeClassifier(max_depth=1, random_state=42)

try:
    ada_clf = AdaBoostClassifier(
        estimator=stump,
        n_estimators=200,
        learning_rate=0.8,
        random_state=42
    )
except TypeError:
    try:
        ada_clf = AdaBoostClassifier(
            estimator=stump,
            n_estimators=200,
            learning_rate=0.8,
            random_state=42,
            algorithm='SAMME'
        )
    except TypeError:
        ada_clf = AdaBoostClassifier(
            base_estimator=stump,
            n_estimators=200,
            learning_rate=0.8,
            random_state=42,
            algorithm='SAMME'
        )

ada_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', ada_clf),
])

print('Training AdaBoost...')
ada_pipeline.fit(X_train, y_train)
print('✅ Model trained successfully!')
"""
            # Need to keep it as a list of lines
            lines = [line + '\n' for line in new_source.split('\n')]
            lines[-1] = lines[-1][:-1] # remove last newline
            cell['source'] = lines

with open('weather_adaboost.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
