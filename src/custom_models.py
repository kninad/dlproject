import numpy as np

import keras
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras import metrics
from keras.layers.normalization import BatchNormalization

ROOT_DIR = '/home/ninad/Desktop/DLproj/code/anomaly/'
CURR_DIR = ROOT_DIR + 'cifar_unct/'

def mlp_model(input_feats_shape, output_classes):
    input_tensor = Input(shape=input_feats_shape, name='inp1')
    flat_tensor = Flatten()(input_tensor)
    x = BatchNormalization(name='post_flat')(flat_tensor)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    logits = Dense(output_classes)(x)
    softmax_output = Activation('softmax', name='softmax_output')(logits)
	
    model = Model(inputs=input_tensor, outputs=softmax_output)    
    return model

def cnn_model(input_feats_shape, output_classes):

    input_tensor = Input(shape=input_feats_shape, name='inp1')
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_tensor)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(512)(x)
    x = Dropout(0.5)(x)
    logits = Dense(output_classes)(x)
    softmax_output = Activation('softmax', name='softmax_output')(logits)
    
    model = Model(inputs=input_tensor, outputs=softmax_output)    
    return model




