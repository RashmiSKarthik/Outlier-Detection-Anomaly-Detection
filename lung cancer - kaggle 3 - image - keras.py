# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 12:09:28 2020

@author: Rashmi S
"""


import os
import numpy as np
np.random.seed(123)
import pandas as pd

from glob import glob
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

import keras.backend as Kb

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, \
                         Flatten, Convolution2D, MaxPooling2D, \
                         BatchNormalization, UpSampling2D
from keras.utils import np_utils

from skimage.io import imread
from sklearn.model_selection import train_test_split
Kb.set_image_data_format('channels_first')
read_image = lambda x: np.expand_dims(imread(x)[::2, ::2],0)
BASE_IMAGE_PATH = os.path.join('F:/implementation', '2d_images')
all_images = glob(os.path.join(BASE_IMAGE_PATH, '*.tif'))
all_masks = ['_masks'.join(c_file.split('_images')) for c_file in all_images]
print(len(all_masks), 'matching files found')


from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
t_img = read_image(all_images[0])
t_msk = read_image(all_masks[0])
fig, (ax1 ,ax2) = plt.subplots(1, 2)
ax1.imshow(t_img[0])
ax2.imshow(t_msk[0])

print('Total samples are', len(all_images))
print('Image resolution is', t_img.shape)

img = np.stack([read_image(i) for i in all_images], 0)
msk = np.stack([read_image(i) for i in all_masks], 0) / 255.0
X_train, X_test, y_train,  y_test = train_test_split(img, msk, test_size=0.1)
print('Training input is', X_train.shape)
print('Training output is {}, min is {}, max is {}'.format(y_train.shape, y_train.min(), y_train.max()))
print('Testing set is', X_test.shape)

#create a deep nn
model = Sequential()
model.add(Convolution2D(filters=32, 
                        kernel_size=(3, 3), 
                        activation='relu', 
                        input_shape=img.shape[1:],
                        padding='same'
                        ))
model.add(Convolution2D(filters=64, 
                        kernel_size=(3, 3), 
                        activation='sigmoid', 
                        input_shape=img.shape[1:],
                        padding='same'
                        ))
model.add(Convolution2D(filters=128, 
                        kernel_size=(3, 3), 
                        activation='sigmoid', 
                        input_shape=img.shape[1:],
                        padding='same'
                        ))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dense(128, activation='relu'))
model.add(Convolution2D(filters=1, 
                        kernel_size=(3, 3), 
                        activation='sigmoid', 
                        input_shape=img.shape[1:],
                        padding='same'
                        ))

model.add(UpSampling2D(size=(2,2)))
model.compile(loss='binary_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy','mse'])
print(model.summary())

history = model.fit(X_train, y_train, validation_split=0.10, epochs=30, batch_size=30)
history.history.keys()
fig, ax = plt.subplots(1,2)
ax[0].plot(history.history['accuracy'], 'b')
ax[0].set_title('Accuraccy')
ax[1].plot(history.history['loss'], 'r')
ax[1].set_title('Loss')

fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.imshow(X_test[2,0])
ax2.imshow(y_test[2,0])
ax3.imshow(model.predict(X_test)[2,0])
