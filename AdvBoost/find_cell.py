import json

with open('weather_adaboost.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'AdaBoostClassifier' in source:
            print(f"--- Cell index {i} ---")
            print(source.encode('ascii', 'ignore').decode('ascii'))

