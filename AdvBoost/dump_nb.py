import json

with open('weather_adaboost.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('cells.txt', 'w', encoding='utf-8') as f:
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            f.write(f"--- CELL {i} ---\n")
            f.write(''.join(cell['source']) + '\n\n')
