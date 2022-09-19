import matplotlib.pyplot as plt

def plot_performance(model_history):
    # model performance visualization
    # subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # accuracy
    ax1.plot(model_history.history['accuracy'], color='red')
    ax1.plot(model_history.history['val_accuracy'], color='green')
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['train', 'validation'], loc='lower right')

    # "Loss"
    ax2.plot(model_history.history['loss'], color='red')
    ax2.plot(model_history.history['val_loss'], color='green')
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['train', 'validation'], loc='upper right')
    plt.show()
