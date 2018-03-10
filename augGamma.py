
import numpy as np
import cv2

from keras.datasets import cifar10

ROOT = '/home/ninad/Desktop/DLproj/'
catgs = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
          'horse', 'ship', 'truck']

(x_test, y_test) = cifar10.load_data()[1]
print('Data is loaded\n')

num = 1000 #to test dropout on
T_mc = 100
shuff_idxs = np.load(ROOT+'code/keras_cifar10/rand_idxs.npy')
rand_idxs = shuff_idxs[:num] #take the 1st num examples
xdata = x_test[rand_idxs, :,:,:]

#images should be in 'uint8' format
def augment_gamma(images, gamma=1.0):
  # build a lookup table mapping the pixel values [0, 255] to
  # their adjusted gamma values
  invGamma = 1.0 / gamma
  table = np.array([((i / 255.0) ** invGamma) * 255 
                    for i in np.arange(0, 256)]).astype("uint8")
  
  # apply gamma correction using the lookup table
  return np.asarray([cv2.LUT(img, table) for img in images])

g1 = 0.6
g2 = 2.1

xgam1 = augment_gamma(xdata, g1)
xgam1 = xgam1.astype(np.float64)
xgam1 /= 255
np.save(ROOT + './code/keras_cifar10/xgam1.npy', xgam1)

xgam2 = augment_gamma(xdata, g2)
xgam2 = xgam2.astype(np.float64)
xgam2 /= 255
np.save(ROOT + './code/keras_cifar10/xgam2.npy', xgam2)








