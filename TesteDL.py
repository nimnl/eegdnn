# -*- coding: utf-8 -*-

import numpy as np

import scipy as sp

from scipy.signal import butter, lfilter
from scipy.signal import freqz

import sklearn as sk
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import h5py
import csv
import matplotlib.pyplot as plt

import sys
#import cPickle
import pdb # para debugar
import networkx as NX
#np.random.seed(123)  # for reproducibility
#from pyhdf import SD, SDC

from scipy import io
#from ffnet import ffnet, mlgraph, savenet, loadnet, exportnet
from scipy.signal import butter, lfilter
from scipy.signal import freqz

import os, sys
from os import listdir
from os.path import isfile, join
from sklearn import preprocessing


import math
#####################
import keras
from keras import callbacks
from keras.callbacks import TensorBoard
from keras.constraints import maxnorm
from keras.models import load_model
from keras import regularizers
from operator import itemgetter
from keras.callbacks import Callback

#####################################################
from keras import backend as K
K.set_image_dim_ordering('th') 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, UpSampling3D, Convolution2D, MaxPooling2D, Convolution1D, AveragePooling2D, Conv1D, MaxPooling1D, AveragePooling1D, UpSampling1D
from keras.models import Model
from keras.utils import np_utils
####################################################

import preprocess #modulo preprocess.py deve estar na mesma pasta

TF_CPP_MIN_LOG_LEVEL=2 #filter out warnings regarding memory usage

#####################################################
def trainConvolutionalAutoEncoder2HiddenLayers(X_train, X_test, nEpochs, batchSize, modelNameToSave, weightNameToSave, nChannels, windW):
    print ('Training conv net...')
    # 5. Preprocess input data
    X_train = X_train.reshape(X_train.shape[0], 1, nChannels, windW)
    X_test = X_test.reshape(X_test.shape[0], 1, nChannels, windW)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    noise_factor = 0.25
    x_train_noisy = X_train #+ noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    x_test_noisy = X_test #+ noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

    # Set input dimensions:

    input_img = Input(shape=(1, nChannels, windW))

    convSize=8
    nFilters = 8
    x = Convolution2D(nFilters, 1, convSize, activation='sigmoid', border_mode='same', name='conv1', activity_regularizer=regularizers.l1(0.01))(input_img)
    x = MaxPooling2D((1, 2), border_mode='same', name='mxp1')(x)
    ##
    x = Convolution2D(nFilters, 1, int(convSize/2), activation='sigmoid', border_mode='same', name='conv2', activity_regularizer=regularizers.l1(0.01))(x)
    x = MaxPooling2D((1, 2), border_mode='same', name='mxp2')(x)
    ## emenda do funil
    e = Convolution2D(nFilters, 1, int(convSize/2), activation='sigmoid', border_mode='same', name='conv2', activity_regularizer=regularizers.l1(0.01))(x)
    x = UpSampling2D((1, 2), name='ups1')(x)
    ##
    x = Convolution2D(nFilters, 1, convSize, activation='sigmoid', border_mode='same', name='conv3', activity_regularizer=regularizers.l1(0.01))(x)
    x = UpSampling2D((1, 2), name='ups2')(x)
    #expected conv11 to have shape (None, 64, 7, 1920) but got array with shape (2403, 1, 7, 7680)
    decoded = Convolution2D(1, 1, convSize, activation='sigmoid', border_mode='same', name='conv4', activity_regularizer=regularizers.l1(0.01))(x)

    autoencoder = Model(input_img, decoded)

    #principal diferenca da vesao 6 para esta versao 7 eh o optimizer
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    autoencoder.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    #autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    earlyStopping=callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
    
    history = autoencoder.fit(x_train_noisy, X_train, nb_epoch=nEpochs, batch_size=batchSize, shuffle=True, validation_data=(x_test_noisy, X_test), verbose=True, callbacks=[earlyStopping])

    autoencoder.save(modelNameToSave)  # creates a HDF5 file 'my_model.h5'
    autoencoder.save_weights(weightNameToSave)  # creates a HDF5 file 'my_model.h5'
    del autoencoder  # deletes the existing model
########################################################
########################################################

#if __name__ == '__main__':

files =  ['s01.dat', 's02.dat', 's03.dat', 's04.dat']#, 's05.dat']
"""
, 's06.dat', 
           's07.dat', 's08.dat', 's09.dat', 's10.dat', 's11.dat', 's12.dat', 
           's13.dat', 's14.dat', 's15.dat', 's16.dat', 's17.dat', 's18.dat', 
           's19.dat', 's20.dat', 's21.dat', 's22.dat', 's23.dat', 's24.dat', 
           's25.dat', 's26.dat', 's27.dat', 's28.dat', 's29.dat', 's30.dat',
           's31.dat', 's32.dat']
"""

data_dims = np.shape(preprocess.load_file(files[0])[0])

#concatenaremos todos os participantes em um único vetor data
data_dims = (len(files)*data_dims[0], data_dims[1], data_dims[2])
data = np.zeros(data_dims)

print("Loading files...")
for file in range(len(files)):
    data[file*40:(file+1)*40] = preprocess.load_file(files[file])[0]

seed = 1 #semente para embaralhar os videos

data, ordem_vids_shuf = preprocess.preprocessData(data, seed)

X_Train = data[0:(len(files)-2)*40] #30 primeiras pessoas (1200 primeiros vídeos)
X_Dev = data[(len(files)-2)*40:(len(files)-1)*40] #31a pessoa
X_Test = data[(len(files)-1)*40:(len(files))*40] #32a pessoa

nEpochs = 10
batchSize = 32
modelNameToSave = 'deepNetwork1.net'
weightNameToSave = 'deepWeights1.net'
nChannels = 40
windW = 7680#8064

trainConvolutionalAutoEncoder2HiddenLayers(X_Train, X_Train, nEpochs, batchSize, modelNameToSave, weightNameToSave, nChannels, windW)


#print(np.shape(data))



























