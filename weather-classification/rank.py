import sys
import os
import json

accuracies = list()
SAVE_DIR = sys.argv[1]
for model_dir in os.listdir(SAVE_DIR):
    if '_' not in model_dir:
        continue

    filenames = os.listdir(os.path.join(SAVE_DIR, model_dir))
    if 'evaluation.txt' in filenames and 'history.txt' in filenames:
        temp = list()

        # Validation accuracy
        history = json.load(open(os.path.join(SAVE_DIR, model_dir, 'history.txt'), 'r'))
        temp.append(round(max(history['val_accuracy'].values()), 4))

        # Test accuracy
        with open(os.path.join(SAVE_DIR, model_dir, 'evaluation.txt'), 'r') as f:
            contents = f.read().splitlines()
        temp.append(round(float(contents[-1].split(' ')[-1]), 4))

        temp.append(model_dir)
    accuracies.append(temp)

# Sort the items by the sum of val_accuracy and test_accuracy
accuracies = list(sorted(accuracies, key=lambda l: l[0] + l[1], reverse=True))
for accuracy in accuracies:
    print(accuracy)