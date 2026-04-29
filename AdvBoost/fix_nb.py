import json

with open('weather_adaboost.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the feature engineering cell we inserted
fe_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "# Feature Engineering" in source and "Temp_Wind" in source:
            fe_idx = i
            break

# We only need the first 7 lines (up to numeric_feature_cols = ...)
if fe_idx != -1:
    source = nb['cells'][fe_idx]['source']
    # Keep only up to line 7
    new_source = [line for line in source if not line.startswith('# Re-create preprocessor') and not line.startswith('preprocessor =') and not line.startswith("    ('num'") and not line.startswith("    ('cat'") and not line.strip() == "])"]
    nb['cells'][fe_idx]['source'] = new_source

with open('weather_adaboost.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Fixed notebook.")
