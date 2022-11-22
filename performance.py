import matplotlib.pyplot as plt

def plot_performance(model_history, title):
    '''
    Outputs the accuracy and loss graphs for the training history
    :param model_history: the history of the model
    :return: void, saves the plot as myPlot.png
    '''

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # accuracy
    ax1.plot(model_history.history['accuracy'], color='blue')
    ax1.plot(model_history.history['val_accuracy'], color='orange')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['train', 'validation'], loc='lower right')

    # loss
    ax2.plot(model_history.history['loss'], color='blue')
    ax2.plot(model_history.history['val_loss'], color='orange')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['train', 'validation'], loc='upper right')
    # plt.show()
    plt.savefig(title)