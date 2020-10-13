#
# Project 1, starter code part a
#

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

NUM_CLASSES = 3

epochs = 1000
batch_size = 32
num_neurons = 10
seed = 10

np.random.seed(seed)
tf.random.set_seed(seed)

histories = {}

#read train data
train_input = np.genfromtxt('ctg_data_cleaned.csv', delimiter= ',')
trainX, train_Y = train_input[1:, :21], train_input[1:,-1].astype(int)
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))
trainY = train_Y-1

# experiment with small datasets
trainX = trainX[:1000]
trainY = trainY[:1000]

# create the model
starter_model = keras.Sequential([
    keras.layers.Dense(num_neurons, activation='relu'),
    keras.layers.Dense(NUM_CLASSES)
])

starter_model.compile(optimizer='sgd',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train the model
histories['starter'] =starter_model.fit(trainX, trainY,
                                        epochs=epochs,
                                        verbose = 2,
                                        batch_size=batch_size)

# plot learning curves
plt.plot(histories['starter'].history['accuracy'], label=' starter model training accuracy')
plt.ylabel('Train accuracy')
plt.xlabel('No. epoch')
plt.legend(loc="lower right")
plt.show()

