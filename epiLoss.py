
import numpy as np

import keras
from keras.datasets import cifar10
from keras.models import load_model

ROOT = '/home/ninad/Desktop/DLproj/'
catgs = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
          'horse', 'ship', 'truck']


# prob - prediction probability for each class(C). Shape: (N, C)
# returns - Shape: (N)
def predictive_entropy(prob):
    eps = 1e-12
    return -1 * np.sum(np.log(prob+eps) * prob, axis=1)

# model - the trained classifier(C classes) where the last layer applies softmax
# X_data - a list of input data(size N)
# T - the number of monte carlo simulations to run
def montecarlo_prediction(model, X_data, T):
	# shape: (T, N, C)
	predictions = np.array([model.predict(X_data) for _ in range(T)])

	# shape: (N, C)
	prediction_probabilities = np.mean(predictions, axis=0)
	
	# shape: (N)
	prediction_variances = predictive_entropy(prediction_probabilities)
	return (prediction_probabilities, prediction_variances)

#~ (x_train, y_train), (x_test, y_test) = cifar10.load_data()
(x_test, y_test) = cifar10.load_data()[1]
print('Data is loaded\n')

#~ x_train = x_train.astype(np.float64)
#~ x_train /= 255.0
#~ y_train = keras.utils.to_categorical(y_train, num_classes)

x_test = x_test.astype(np.float64)
x_test /= 255.0
numClass=10
y_test = keras.utils.to_categorical(y_test, numClass)

model = load_model(ROOT + 'code/keras_cifar10/cifar10_trained_model_0.8.h5')

#~ N_test = x_test.shape[0]
#~ all_idxs = np.arange(N_test)
#~ np.random.shuffle(all_idxs) # in place shuffling
num = 1000 #to test dropout on
T_mc = 100
shuff_idxs = np.load(ROOT+'code/keras_cifar10/rand_idxs.npy')
rand_idxs = shuff_idxs[:num] #take the 1st num examples
xdata = x_test[rand_idxs, :,:,:]
ydata = y_test[rand_idxs]

(predProb, predVar) = montecarlo_prediction(model, xdata.transpose(0,3,1,2), T_mc)
np.save(ROOT+'code/keras_cifar10/pprob1000.npy', predProb)
np.save(ROOT+'code/keras_cifar10/pvar1000.npy', predVar) 






