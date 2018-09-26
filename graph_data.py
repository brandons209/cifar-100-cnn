import pickle
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

#Takes a path to a pickle file and returns it's content
def load_network_data(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

#takes all data in a directory path and returns a list containing the data in each file.
def load_all_data(path):
    raw_data = []
    for file in sorted(glob(path)):
        raw_data.append(load_network_data(file))
    return raw_data

#plots a single graph
def plot_graph(x, y, x_label, y_label, label, title, markersize):
    plt.title(title)
    plt.plot(x, y, linewidth=2, label=label, marker='o', markersize=markersize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)

#unused, here for reference
"""
def plot_data(raw_data, legend_title, time_x_title, time_x_range, model_or_data_size):
    sizes_metric = []
    losses = []
    val_losses = []
    time_taken = []
    for run in raw_data:#extracts metrics from the raw data
        losses.append(run[0]['loss'])
        val_losses.append(run[0]['val_loss'])
        time_taken.append(run[1]['train_time']/60)
        if model_or_data_size:
            sizes_metric.append(np.round(run[2]['params']*(10**-6), 3))
        else:
            sizes_metric.append(run[2]['params'])

    epochs = len(losses[0])
    index_epochs = np.arange(epochs) + 1

    plt.figure(figsize=(10,10))

    for i, size in enumerate(sizes_metric):#plot number of different runs
        plt.subplot(3, 1, 1)
        plot_graph(index_epochs, losses[i], "Epochs", "Training Loss", str(sizes_metric[i]), "Training Loss vs. Epochs")
        plt.xticks(index_epochs)
        plt.yticks(np.linspace(0, 5, num=10))
        plt.legend(title=legend_title)

        plt.subplot(3, 1, 2)
        plot_graph(index_epochs, val_losses[i], "Epochs", "Validation Loss", str(sizes_metric[i]), "Validation Loss vs. Epochs")
        plt.xticks(index_epochs)
        plt.yticks(np.linspace(0, 5, num=10))
        plt.legend(title=legend_title)

    plt.subplot(3, 1, 3)
    plot_graph(sizes_metric, time_taken, time_x_title, "Train Time (minutes)", None, "Size vs. Time")
    plt.xticks(time_x_range)
    plt.yticks(np.arange(0, 40, step=2))
    plt.legend()

    plt.tight_layout()
    plt.show()
"""
#plots all data in given raw_data, raw_data is list containing the data for each trained model based on different sizes metrics
#creates a new figure for each plot.
def plot_data(raw_data, legend_title, time_x_title, time_x_range, model_or_data_size):
    sizes_metric = []
    top_5_train_accuracy = []
    top_5_valid_accuracy = []
    val_losses = []
    time_taken = []
    for run in raw_data:#extracts metrics from the raw data and appends them to lists
        top_5_train_accuracy.append(run[0]['sparse_top_k_categorical_accuracy'])
        top_5_valid_accuracy.append(run[0]['val_sparse_top_k_categorical_accuracy'])
        val_losses.append(run[0]['val_loss'])
        time_taken.append(run[1]['train_time']/60)
        if model_or_data_size:
            sizes_metric.append(np.round(run[2]['params']*(10**-6), 3))
        else:
            sizes_metric.append(run[2]['params'])

    epochs = len(top_5_train_accuracy[0])
    index_epochs = np.arange(epochs) + 1

    plt.figure(figsize=(10,10))
    for i, size in enumerate(sizes_metric):#plot Top 5 training accuracy for all runs
        plot_graph(index_epochs, top_5_train_accuracy[i], "Epochs", "Top 5 Training Accuracy", str(sizes_metric[i]), "Top 5 Training Accuracy vs. Epochs", 3)
        plt.xticks(index_epochs)
        plt.yticks(np.linspace(0, 1, num=10))
        plt.legend(title=legend_title)

    plt.show()

    plt.figure(figsize=(10,10))
    for i, size in enumerate(sizes_metric):#plot top 5 validation accuracy for all runs
        plot_graph(index_epochs, top_5_valid_accuracy[i], "Epochs", "Top 5 Validation Accuracy", str(sizes_metric[i]), "Top 5 Validation Accuracy vs. Epochs", 3)
        plt.xticks(index_epochs)
        plt.yticks(np.linspace(0, 1, num=10))
        plt.legend(title=legend_title)

    plt.show()

    plt.figure(figsize=(10,10))
    for i, size in enumerate(sizes_metric):#plot validation loss for all runs
        plot_graph(index_epochs, val_losses[i], "Epochs", "Validation Loss", str(sizes_metric[i]), "Validation Loss vs. Epochs", 3)
        plt.xticks(index_epochs)
        plt.yticks(np.linspace(0, 5, num=10))
        plt.legend(title=legend_title)

    plt.show()

    plt.figure(figsize=(10,10))
    for i, size in enumerate(sizes_metric):#plots last point for each run for validation loss in an easier to see graph
        plot_graph(index_epochs[28:], val_losses[i][28:], "Epochs", "Validation Loss", str(sizes_metric[i]), "Validation Loss vs. Epochs", 8)
        plt.xticks(index_epochs[28:])
        plt.yticks(np.linspace(2, 2.25, num=30))
        plt.legend(title=legend_title)

    plt.show()

    plt.figure(figsize=(10,10))#plots the size metric vs time graph
    plot_graph(sizes_metric, time_taken, time_x_title, "Train Time (minutes)", None, "Size vs. Time", 6)
    plt.xticks(time_x_range)
    plt.yticks(np.arange(0, 40, step=2))
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_data(load_all_data('saved_history_data/largest-dataset-size/*'), "Parameters in millions",
          "Model Size (no. of parameters in millions)", np.linspace(7.5, 8, num=5), True)

plot_data(load_all_data('saved_history_data/largest-network-size/*'), "Dataset Sizes", "Dataset Size", np.arange(9000, 46000, step=1000), False)
