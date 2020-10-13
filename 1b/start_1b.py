#
# Project 1, starter code part b
#

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

NUM_CLASSES = 7

epochs = 300
batch_size = 8
num_neurons = 30
seed = 10

histories={}

np.random.seed(seed)
tf.random.set_seed(seed)

#read and divide data into test and train sets 
admit_data = np.genfromtxt('admission_predict.csv', delimiter= ',')
X_data, Y_data = admit_data[1:,1:8], admit_data[1:,-1]
Y_data = Y_data.reshape(Y_data.shape[0], 1)

idx = np.arange(X_data.shape[0])
np.random.shuffle(idx)
X_data, Y_data = X_data[idx], Y_data[idx]

# experiment with small datasets
trainX = X_data[:100]
trainY = Y_data[:100]

trainX = (trainX- np.mean(trainX, axis=0))/ np.std(trainX, axis=0)

# create a network
starter_model = keras.Sequential([
    keras.layers.Dense(num_neurons, activation='relu'),
    keras.layers.Dense(1)
])

starter_model.compile(optimizer='sgd',
              loss=keras.losses.MeanSquaredError(),
              metrics=['mse'])

# learn the network
histories['starter'] =starter_model.fit(trainX, trainY,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        verbose = 2)

# plot learning curves
plt.plot(histories['starter'].history['mse'], label=' starter model training mse')
plt.ylabel('mse')
plt.xlabel('No. epoch')
plt.legend(loc="lower right")
plt.show()

