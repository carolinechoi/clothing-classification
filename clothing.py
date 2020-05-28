# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Dataset Info
print(train_images.shape)
print(test_images.shape)
# each label is an integer between 0 and 9 (stored in an array)
# each image represented as 28x28 pixels

# Preprocess data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# scale values to a range of 0 to 1 before feeding them to the neutral network model
# divide values by 255 (that way min is 0, max is 1) --> notice that range of plot is from 0 to around 255

train_images = train_images/255.0 # make it a double!
test_images = test_images/255.0

# check to see if images are formatted
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Build layer
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # transforms format of images from 2D to 1D array
    keras.layers.Dense(128, activation='relu'), # 128 nodes (neurons)
    keras.layers.Dense(10) # returns array w/ length of 10, each node contains a score that belongs to one of the 10 classes
])

model.compile(optimizer='adam',  # how the model is updated based on data is sees/loss function
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # measures how accurate the model is during training
              metrics=['accuracy']) # used to monitor the training and testing steps (accuracy measured here)

model.fit(train_images, train_labels, epochs=10) # fits model to raining data
                                                 # matches images to labels

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)