# CIFAR100 Convolutional Neural Network
## Identifying pictures of 100 different objects.
***
This CNN model uses the CIFAR100 data-set, a set containing 60,000 images of 100 different objects, each class containing about 500 different images.

[LICENSE](LICENSE)
***

### Libraries
I used the Keras deep learning library with the TensorFlow back end, and of course numpy. I trained this model on my NVIDIA 1060 with 6GB of VRAM.

### Model Architecture
This model consists of 3 convolutional layers followed by a max pooling layer, then another 4 convolutional layers followed by another max pooling layer. Padding for all the layers was set to "same", which means they were zero-padded. The classifier layers has one dense hidden layer followed by the output layer to predict the classes with probabilities.    

Dropout was added after the first block, after the fourth convolution layer in the second block, after the second block, and after the first hidden layer in the classifier dense layer to help with over fitting.    

Activation functions of all layers was the ReLu function, except the last output layer, which had a softmax activation for probability predictions.

### Running the Model
1. Install required packages from requirements.txt file included.
```bash
pip install -r requirements.txt
```
2. Change hyperparameters as needed under the hyperparameters section in cnn.py.
```python
#set hyperparameters
learn_rate = 0.001
epochs = 30
batch_size = 64
```
3. (Optional) If you want to view the graphs for loss and accuracy with TensorBoard, run:
```bash
tensorboard --logdir=tensorboard_logs
```

4. Run cnn.py:
```bash
python cnn.py
```

### Performance
I was able to score 47% validation and testing accuracy when training this model on 45,000 images. It only took about 40 minutes to train. Being able to predict between 100 different objects on 40 minutes of training on this relatively small network with nearly 50% accuracy is a great start to this model structure. I suspect that that taking 5000 images from testing and put it with the original 50,000 images for the training, leaving 55,000 images for training, 2500 for validation and testing sets, while also adding 3 to 4 more "blocks" of convolutional and max pooling layers that I can push 60-65% accuracy.
