import numpy as np
from sklearn import linear_model, metrics
from sklearn.metrics import roc_auc_score, average_precision_score
import utils

ROOT_DIR = '/home/ninad/Desktop/DLproj/code/anomaly/'
CURR_DIR = ROOT_DIR + 'cifar_unct/'

expName = 'cifar10' # mnist or cifar10
labelsIn = [0,1,2,3,4,5,6]

linTup, dnnTup = utils.get_data(expName, labelsIn)
(x_lin_train, y_lin_train), (x_lin_test, y_lin_test) = linTup
(x_nn_train, y_nn_train), (x_nn_test, y_nn_test) = dnnTup
print('data loaded\n')

lin_Xtrn = np.load(CURR_DIR + expName + '_lin_Xtrn.npy')
lin_Xtst = np.load(CURR_DIR + expName + '_lin_Xtst.npy')
lr = linear_model.LogisticRegressionCV(Cs=100, scoring='roc_auc').fit(
                                        lin_Xtrn, y_lin_train)

y_pred = lr.predict(lin_Xtst)
score_aps = average_precision_score(y_lin_test, y_pred)
score_roc = roc_auc_score(y_lin_test, y_pred)
print(score_aps, score_roc, '\n')

lin_Xtrn_normal = np.load(CURR_DIR + expName + '_lin_Xtrn_normal.npy')
lin_Xtrn_normal = lin_Xtrn_normal.reshape(-1,1)
lin_Xtst_normal = np.load(CURR_DIR + expName + '_lin_Xtst_normal.npy')
lin_Xtst_normal = lin_Xtst_normal.reshape(-1,1)
lr_normal = linear_model.LogisticRegressionCV(Cs=100, scoring='roc_auc').fit(
                                        lin_Xtrn_normal, y_lin_train)

y_pred_normal = lr_normal.predict(lin_Xtst_normal)
score_aps_normal = average_precision_score(y_lin_test, y_pred_normal)
score_roc_normal = roc_auc_score(y_lin_test, y_pred_normal)
print(score_aps_normal, score_roc_normal, '\n')




