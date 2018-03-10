import numpy as np
import keras.models
import utils

ROOT_DIR = '/home/ninad/Desktop/DLproj/code/anomaly/'
CURR_DIR = ROOT_DIR + 'cifar_unct/'

expName = 'cifar10' # mnist or cifar10
labelsIn = [0,1,2,3,4,5,6]

linTup, dnnTup = utils.get_data(expName, labelsIn)
(x_lin_train, y_lin_train), (x_lin_test, y_lin_test) = linTup
(x_nn_train, y_nn_train), (x_nn_test, y_nn_test) = dnnTup
print('data loaded\n')

mPath = CURR_DIR + expName + '_mod01.h5'
model = keras.models.load_model(mPath)

lin_Xtrn = utils.get_feats(model, x_lin_train, T=100)
lin_Xtst = utils.get_feats(model, x_lin_test, T=100)

np.save(CURR_DIR + expName + '_lin_Xtrn.npy', lin_Xtrn)
np.save(CURR_DIR + expName + '_lin_Xtst.npy', lin_Xtst)

# FEATURES FOR THE NORMAL NETWORK
lin_Xtrn_normal = utils.get_feats_normal(model, x_lin_train)
lin_Xtst_normal = utils.get_feats_normal(model, x_lin_test)

np.save(CURR_DIR + expName + '_lin_Xtrn_normal.npy', lin_Xtrn_normal)
np.save(CURR_DIR + expName + '_lin_Xtst_normal.npy', lin_Xtst_normal)


