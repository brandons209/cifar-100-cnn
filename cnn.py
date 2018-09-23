#general imports
import numpy as np
import helper as help
import time

#data set import
from keras.datasets import cifar100

#model imports
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

#training
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import optimizers as opt

#load cifar100 images with fine labels.
(train_images, train_labels), (test_images, test_labels) = cifar100.load_data(label_mode='fine')
#take last 5000 images and labels for validation set
valid_images, valid_labels = train_images[:5000], train_labels[:5000]
#remove validation set from training data
train_images, train_labels = train_images[5000:], train_labels[5000:]

#normalize pixel values from [0, 255] to [0,1] for better training and predicting
train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255
valid_images = valid_images.astype('float32')/255

print("Data Stats:")
print("Training images shape: {}".format(train_images.shape))
print("There are {} training images, {} testing images, and {} validation images.".format(len(train_images), len(test_images), len(valid_images)))

#build model
cnn = Sequential()

"""
This model has two "blocks" consisting of two convolutional layers followed by a max pool layer.
The classifier layers has one hidden layer then the output layer.
"""
cnn.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu', input_shape=(train_images.shape[1:])))
cnn.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=2, padding='same'))

cnn.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
cnn.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=2, padding='same'))
cnn.add(Dropout(0.3))

cnn.add(Flatten())
cnn.add(Dense(756, activation='relu'))
cnn.add(Dropout(0.3))
cnn.add(Dense(100, activation='softmax'))

cnn.summary()
input("Press enter to begin training of model...")

weight_save_path = 'cnn.best.weights.hdf5'
checkpointer = ModelCheckpoint(filepath=weight_save_path, verbose=1, save_best_only=True)
start_time = time.strftime("%a_%b_%d_%Y_%H:%M", time.localtime())
ten_board = TensorBoard(log_dir='tensorboard_logs/{}_cnn'.format(start_time), write_images=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=0.00000000001)

#set gyperparameters
learn_rate = 0.001
epochs = 20
batch_size = 64

#I use sparse_categorical_crossentropy loss function since it works like categorical_crossentropy but the labels do not have to be one-hot encoded, saving computation time.
cnn.compile(loss='sparse_categorical_crossentropy', optimizer=opt.RMSprop(lr=learnrate), metrics=['accuracy'])

#train model:
print("TensorBoard logs are viewable in the tensorboard_logs directory, this log folder will be written as: {}".format(start_time + '_cnn'))
print("Best weights will be saved to: {}".format(weight_save_path))
time.sleep(5)

cnn.fit(train_images, train_labels, validation_data=(valid_images, valid_labels), epochs=epochs,
        batch_size=batch_size, callbacks=[checkpointer, ten_board, reduce_lr], verbose=1)

input("Press enter to test model...")

metrics = cnn.evaluate(test_images, test_labels)
print("Testing loss: {:.4f}, Testing accuracy: {:.2f}".format(metrics[0], metrics[1]))
