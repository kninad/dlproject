#~ import tensorflow as tf
#~ from keras.backend.tensorflow_backend import set_session
#~ config = tf.ConfigProto()
#~ #config.gpu_options.per_process_gpu_memory_fraction = 0.8
#~ config.gpu_options.visible_device_list = "0"
#~ config.gpu_options.allow_growth = True
#~ set_session(tf.Session(config=config))

import numpy as np

import keras
from keras.datasets import cifar10
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras import metrics

from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.engine.topology import Layer
from keras.layers.wrappers import TimeDistributed
from keras.utils.generic_utils import get_custom_objects

from aleoLoss import *

ROOT = '/home/ninad/Desktop/DLproj/'
num_classes = 10

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('Data is loaded\n')
x_train = x_train.astype(np.float64)
x_test = x_test.astype(np.float64)
x_train /= 255.0
x_test /= 255.0
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def bayesian_model(input_feats_shape, output_classes):

    input_tensor = Input(shape=input_feats_shape, name='inp1')

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_tensor)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(512)(x)
    x = Dropout(0.5)(x)

    logits = Dense(output_classes)(x)
    variance_pre = Dense(1)(x)
    variance = Activation('softplus', name='variance')(variance_pre)
    
    logits_variance = concatenate([logits, variance], name='logits_variance')
    softmax_output = Activation('softmax', name='softmax_output')(logits)

    model = Model(inputs=input_tensor, outputs=[logits_variance,softmax_output])
    return model

#~ inpFeatShape = x_train.shape[1:]
#~ numEpochs = 10
#~ batchSize = 32
#~ model = bayesian_model(inpFeatShape, num_classes)

#~ m1path = './wB40.h5'
#~ m2path='./wB50.h5'
#~ model.load_weights(m1path)

#~ model.compile(
	#~ optimizer=Adam(lr=1e-3, decay=0.001),
	#~ loss={
	#~ 'logits_variance': bayesian_categorical_crossentropy(100, 10),    
    #~ 'softmax_output': 'categorical_crossentropy'},
    #~ metrics={'softmax_output': metrics.categorical_accuracy},
    #~ loss_weights={'logits_variance': .2, 'softmax_output': 1.})


#~ model.fit(x_train,
    #~ {'logits_variance':y_train, 'softmax_output':y_train},  
    #~ epochs=numEpochs, 
    #~ batch_size=batchSize)

#~ print('Finished training.\n')

#~ # Save model and weights
#~ #model.save(model_path, custom_objects={'logits_variance': bayesian_categorical_crossentropy})
#~ model.save_weights(m2path)
#~ print('Saved trained model weights at %s \n' % m2path)




#~ opts1 = model.predict(xgam1)
#~ y1 = opts1[0]
#~ y2 = opts1[1]
#~ altvar_g1 = np.asarray([ y1[i,:][-1] for i in range(y1.shape[0])])
#~ altlogits_g1 = np.asarray([y1[i,:][:-1] for i in range(y1.shape[0])])
#~ np.save('./atlvar_g1.npy', altvar_g1)
#~ np.save('./altlogits_g1.npy', altlogits_g1)
#~ np.save('./altprob_g1.npy', y2)


#~ opts2 = model.predict(xgam2)
#~ w1 = opts2[0]
#~ w2 = opts2[1]
#~ altvar_g2 = np.asarray([ w1[i,:][-1] for i in range(w1.shape[0])])
#~ altlogits_g2 = np.asarray([w1[i,:][:-1] for i in range(w1.shape[0])])
#~ np.save('./atlvar_g2.npy', altvar_g2)
#~ np.save('./altlogits_g2.npy', altlogits_g2)
#~ np.save('./altprob_g2.npy', w2)






#Evaluation
#~ scores = model.evaluate(x_test, 
    #~ {'logits_variance': y_test, 'softmax_output':y_test},
    #~ verbos=1)
#~ print('Test loss:\n', scores[0])
#~ print('Test accuracy:\n', scores[1])













