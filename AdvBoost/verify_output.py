import json
import sys

with open('weather_adaboost.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

errors = []
images = 0
accuracy = None

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'error':
                errors.append(output)
            if 'data' in output and 'image/png' in output['data']:
                images += 1
            if output.get('output_type') == 'stream':
                text = "".join(output['text'])
                if 'accuracy' in text.lower() or 'score' in text.lower() or 'adaboost' in text.lower():
                    # Let's try to capture the final accuracy.
                    pass

print(f"Errors found: {len(errors)}")
print(f"Images found: {images}")

# We can find the accuracy by looking at the specific cell that prints the accuracy.
# Usually it prints something like "AdaBoost Accuracy: 0.85" or similar.
for cell in reversed(nb['cells']):
    if cell['cell_type'] == 'code':
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'stream':
                text = "".join(output.get('text', []))
                for line in text.split('\n'):
                    if 'accuracy' in line.lower() and '%' in line:
                        print("FOUND ACCURACY LINE:", line)
                    elif 'accuracy' in line.lower() and '.' in line:
                        print("FOUND ACCURACY LINE:", line)
