import numpy as np
import helper as help

#model imports
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

#training
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import optimizers as opt

train, test = help.load_data()

print(len(train))
