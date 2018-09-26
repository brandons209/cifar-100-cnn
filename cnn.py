#general imports
import numpy as np
import time
import pickle

#data set import
from keras.datasets import cifar100

#model imports
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras import backend as K

#training
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import optimizers as opt

#load cifar100 images with fine labels.
(train_images, train_labels), (test_images, test_labels) = cifar100.load_data(label_mode='fine')
#flatten labels from (num_samples, 1) to (num_samples,), this is for the sparse_top_k_categorical_accuracy metric
train_labels, test_labels = train_labels.flatten(), test_labels.flatten()
#take last 5000 images and labels for validation set
valid_images, valid_labels = train_images[:5000], train_labels[:5000]
#remove validation set from training data
train_images, train_labels = train_images[5000:], train_labels[5000:]
#change size of training dataset
#train_images, train_labels = train_images[:36000], train_labels[:36000]

#normalize pixel values from [0, 255] to [0,1] for better training and predicting
train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255
valid_images = valid_images.astype('float32')/255

#print dataset stats
print("Data Stats:")
print("Training images shape: {}".format(train_images.shape))
print("There are {} training images, {} testing images, and {} validation images.".format(len(train_images), len(test_images), len(valid_images)))

#build model
cnn = Sequential()

"""
This model has two "blocks" consisting of two convolutional layers with same padding followed by a max pooling layer.
The classifier has one hidden layer then the output layer.

Activation for all layers, except the output layer, is the ReLu function, has that function seems to work best for CNNs
The output layers uses a softmax activation function for probalities.

For each increase in model depth, I added another convolutional layer to the first block, then added one to the second block twice, along with dropout layers.
"""
cnn.add(Conv2D(filters=200, kernel_size=2, padding='same', activation='relu', input_shape=(train_images.shape[1:])))
cnn.add(Conv2D(filters=200, kernel_size=2, padding='same', activation='relu'))
cnn.add(Conv2D(filters=200, kernel_size=2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=2, padding='same'))
cnn.add(Dropout(0.4))

cnn.add(Conv2D(filters=200, kernel_size=2, padding='same', activation='relu'))
cnn.add(Conv2D(filters=200, kernel_size=2, padding='same', activation='relu'))
cnn.add(Conv2D(filters=200, kernel_size=2, padding='same', activation='relu'))
cnn.add(Conv2D(filters=200, kernel_size=2, padding='same', activation='relu'))
cnn.add(Dropout(0.4))
cnn.add(MaxPooling2D(pool_size=2, padding='same'))
cnn.add(Dropout(0.5))

cnn.add(Flatten())
cnn.add(Dense(550, activation='relu'))
cnn.add(Dropout(0.4))
cnn.add(Dense(100, activation='softmax'))

cnn.summary()#display summary of model
input("Press enter to begin training of model...")

#checkpointer callback to save weights with best val loss as model trains.
weight_save_path = 'cnn.best.weights.hdf5'
checkpointer = ModelCheckpoint(filepath=weight_save_path, verbose=1, save_best_only=True)
#tensorboard callback to see loss and accuracy stats as model trains:
start_time = time.strftime("%a_%b_%d_%Y_%H:%M", time.localtime())
ten_board = TensorBoard(log_dir='tensorboard_logs/{}_cnn'.format(start_time), write_images=True)
#reduce learning if val_loss does not improve for 2 epochs.
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_lr=1*10**(-11))

#set hyperparameters
learn_rate = 0.001
epochs = 30
batch_size = 64

#I use sparse_categorical_crossentropy loss function since it works like categorical_crossentropy but the labels do not have to be one-hot encoded, saving computation time. I use the Adam optimizer and include metrics for accuracy and sparse_top-5 error as well.
cnn.compile(loss='sparse_categorical_crossentropy', optimizer=opt.Adam(lr=learn_rate), metrics=['accuracy', 'sparse_top_k_categorical_accuracy'])

#train model:
print("TensorBoard logs are viewable in the tensorboard_logs directory, this log folder will be written as: {}".format(start_time + '_cnn'))
print("Best weights will be saved to: {}".format(weight_save_path))
time.sleep(5)
begin_time = time.time()
history = cnn.fit(train_images, train_labels, validation_data=(valid_images, valid_labels), epochs=epochs,
        batch_size=batch_size, callbacks=[checkpointer, ten_board, reduce_lr], verbose=1)
time_to_train = time.time() - begin_time
print("Model took {:.0f} minutes to train.".format(time_to_train/60))
input("Press enter to test model...")

#test model
cnn.load_weights(weight_save_path)
metrics = cnn.evaluate(test_images, test_labels)
print("Testing loss: {:.4f}, Testing accuracy: {:.2f}%, Top 5 Error Accuracy: {:.2f}%".format(metrics[0], metrics[1]*100, metrics[2]*100))

#save history for graphing purposes:
print("Saving data...")
cnn.save('saved_models/test')
#with open('saved_history_data/largest-dataset-size/network_4', 'wb') as file:
#    timedict = {"train_time" : time_to_train}
#    paramsdict = {"params" : int(np.sum([K.count_params(p) for p in set(cnn.trainable_weights)]))}
#    dicts = [history.history, timedict, paramsdict]
#    pickle.dump(dicts, file)
#with open('saved_history_data/largest-network-size/network_5', 'wb') as file:
#    timedict = {"train_time" : time_to_train}
#    paramsdict = {"params" : len(train_images)}
#    dicts = [history.history, timedict, paramsdict]
#    pickle.dump(dicts, file)
