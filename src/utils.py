import matplotlib.pyplot as plt

from src.config import *


def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name, image_name):
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]

    epochs = range(len(metric_value_1))

    plt.plot(epochs, metric_value_1, 'blue', label=metric_name_1)
    plt.plot(epochs, metric_value_2, 'red', label=metric_name_2)

    plt.title(str(plot_name))
    plt.legend()

    plt.savefig("./images/{}".format(str(image_name)), bbox_inches="tight")
    plt.show()
