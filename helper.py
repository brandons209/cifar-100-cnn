import pickle
import numpy as np

def load_data():
    data_dict = 'cifar-100-python/'
    with open(data_dict + 'train', 'rb') as file:
        train = pickle.load(file, encoding='bytes')
    with open(data_dict + 'test', 'rb') as file:
        test = pickle.load(file, encoding='bytes')

    return train, test
