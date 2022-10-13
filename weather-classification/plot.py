import json
import matplotlib.pyplot as plt


history = json.load(open('history.txt', 'r'))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# accuracy
ax1.plot(*zip(*history['accuracy'].items()), color='blue')
ax1.plot(*zip(*history['val_accuracy'].items()), color='red')
ax1.set_title('Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(['train', 'validation'], loc='lower right')
for i, label in enumerate(ax1.xaxis.get_ticklabels()):
    if i % 4 != 0:
        label.set_visible(False)

# loss
ax2.plot(*zip(*history['loss'].items()), color='blue')
ax2.plot(*zip(*history['val_loss'].items()), color='red')
ax2.set_title('Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['train', 'validation'], loc='upper right')
for i, label in enumerate(ax2.xaxis.get_ticklabels()):
    if i % 4 != 0:
        label.set_visible(False)

# fig.tight_layout()
plt.show()
plt.savefig('myPlot')
