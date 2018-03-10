# Data loading and pre-processing file

import numpy as np
import keras

def get_data(dataName, labels_in):
    
    if dataName == 'mnist':
        from keras.datasets import mnist as dataSet
    elif dataName == 'cifar10':
        from keras.datasets import cifar10 as dataSet
    else:
        print('Invalid name for dataset!')
        return
    
    (x_train, y_train), (x_test, y_test) = dataSet.load_data()    
    
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    x_train /= 255.0
    x_test /= 255.0
    
    train_bool = [y_train[i] in labels_in for i in range(len(y_train))]
    test_bool = [y_test[i] in labels_in for i in range(len(y_test))]
    
    x_lin_train = x_train
    y_lin_trian = [int(y_train[i] in labels_in) for i in range(len(y_train))]
    y_lin_trian = np.asarray(y_lin_trian)
    
    x_lin_test = x_test
    y_lin_test = [int(y_test[i] in labels_in) for i in range(len(y_test))]
    y_lin_test = np.asarray(y_lin_test)
    
    x_nn_train = x_train[train_bool]
    x_nn_test = x_test[test_bool]
    
    y_nn_train = y_train[train_bool]
    y_nn_test = y_test[test_bool]    
    
    # Convert class vectors to binary class matrices.
    y_nn_train = keras.utils.to_categorical(y_nn_train, len(labels_in))
    y_nn_test = keras.utils.to_categorical(y_nn_test, len(labels_in))   
    
    linTup = ((x_lin_train, y_lin_trian), (x_lin_test, y_lin_test))
    dnnTup = ((x_nn_train, y_nn_train), (x_nn_test, y_nn_test))
    
    return linTup, dnnTup

def get_normal_data(dataName, num_classes):

    if dataName == 'mnist':
        from keras.datasets import mnist as dataSet
    elif dataName == 'cifar10':
        from keras.datasets import cifar10 as dataSet
    else:
        print('Invalid name for dataset!')
    
    (x_train, y_train), (x_test, y_test) = dataSet.load_data()    
    
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    x_train /= 255.0
    x_test /= 255.0
    
    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)       
    
    return (x_train, y_train), (x_test, y_test)
        
def get_feats(model, X_data, batches=100, T=100):
    N = X_data.shape[0]
    feat_vecs = np.empty((N,3))
    # shape: (T, N, C)
    preds = np.array([model.predict(X_data, batch_size=batches, verbose=1) for _ in range(T)])    
    
    # shape: (N, C)
    mean_probs = np.mean(preds, axis=0)    
	# shape: (N,) #pred_means_ent #A
    feat_vecs[:,0] = -1 * np.sum(np.log(mean_probs) * mean_probs, axis=1) 
    
    # shape: (T,N)
    pred_ents = -1 * np.sum(np.log(preds) * preds, axis=2)
    # shape: (N,)
    feat_vecs[:,1]= np.mean(pred_ents, axis=0) #pred_ents_mean #B
    feat_vecs[:,2]= np.std(pred_ents, axis=0) #pred_ents_std  #C
    
    return feat_vecs

def get_feats_normal(model, X_data, batches=100):
    
    N = X_data.shape[0]
    # shape: (N, C)
    probs = np.array(model.predict(X_data, batch_size=batches, verbose=1))        
	# shape: (N,) #entropy for each data point
    feat_vecs = -1 * np.sum(np.log(probs) * probs, axis=1) 
    
    return feat_vecs

def get_feats_custom(model, X_data, batches=100, T=100):

    N = X_data.shape[0]    
    # shape: (T, N, C)
    preds = np.array([model.predict(X_data, batch_size=batches, verbose=1) for _ in range(T)])    
    return preds
