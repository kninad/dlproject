
import numpy as np

import keras
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras import metrics
from keras.layers.normalization import BatchNormalization

import custom_models
import utils

ROOT_DIR = '/home/ninad/Desktop/DLproj/code/anomaly/'
CURR_DIR = ROOT_DIR + 'cifar_unct/'

expName = 'cifar10'
nof_class = 10
#~ labelsIn = [0,1,2,3,4,5,6]
#~ linTup, dnnTup = utils.get_data(expName, labelsIn)
#~ (x_lin_train, y_lin_trian), (x_lin_test, y_lin_test) = linTup
#~ (x_nn_train, y_nn_train), (x_nn_test, y_nn_test) = dnnTup
(x_nn_train, y_nn_train), (x_nn_test, y_nn_test) = utils.get_normal_data(expName, nof_class)
print('data loaded\n')

inp_shape = x_nn_train.shape[1:]
model = custom_models.cnn_model(inp_shape, nof_class) # for CIFAR10
#model = custom_models.mlp_model(inp_shape, nof_class) # for MNIST
numEpochs = 20 # 30 for cifar10
batchSize = 32

model.compile(optimizer=Adam(lr=1e-3), 
        loss='categorical_crossentropy',
        metrics=[metrics.categorical_accuracy])

print('start training\n')
model.fit(x_nn_train, 
        y_nn_train,  
        epochs=numEpochs, 
        batch_size=batchSize)

# Save model and weights
mPath = CURR_DIR + expName + '_mod_all.h5'
model.save(mPath)
print('Finished training and Saved model at %s \n' % mPath)

model.evaluate(x_nn_test, y_nn_test)
