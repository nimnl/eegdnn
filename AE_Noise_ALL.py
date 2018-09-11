# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import math, time, datetime

from scipy.signal import butter, lfilter
from scipy.signal import freqz

import sklearn as sk
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import h5py

import matplotlib.pyplot as plt

#import cPickle
import pdb # para debugar
import networkx as NX

from scipy import io

import os, sys
from os import listdir
from os.path import isfile, join
from sklearn import preprocessing

#####################
import keras
from keras import callbacks
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from keras.constraints import maxnorm
from keras import regularizers
from operator import itemgetter

from keras.models import load_model
from keras.models import model_from_json

#####################################################
from keras import backend as K
K.set_image_data_format('channels_first') # set format
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, UpSampling3D, Conv2D, MaxPooling2D, Convolution1D, AveragePooling2D, Conv1D, MaxPooling1D, AveragePooling1D, UpSampling1D
from keras.models import Model
from keras.utils import np_utils
####################################################

import preprocess #modulo preprocess.py deve estar na mesma pasta

# o modulo nn_utils.py abaixo tambem deve estar na mesma pasta!!
from nn_utils import (plotData, reshapeData, remapDEAP2, launchGPUMonitor,
                      prepareFilenames, loadData, process_and_addBlink,
                      separateSets, separateSets2, getNetworkNames)
 
#TF_CPP_MIN_LOG_LEVEL=2 #filter out warnings regarding memory usage


#####################################################
#####################################################
#####################################################



def trainConvolutionalAE(X_train, X_test, nEpochs, batchSize, modelNameToSave, weightsNameToSave, nChannels, windW):
    print("Getting start time for training...")
    
    start=time.time()

    print ("Training Conv. AE...")
    

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Set input dimensions:
    input_img = Input(shape=(1, nChannels, windW))

    convSize = 8
    nFilters = 100
    reg = 0#0.01 #regularização

    print ("batchSize: %d | nFilters: %d | %d subjects \n" % (batchSize, nFilters, 
                                                           X_train.shape[0]/40))
    
    x = Conv2D(nFilters, convSize, strides=(1,1), activation='tanh', padding='same',
               name='conv1', activity_regularizer=regularizers.l1(reg))(input_img)
    
    x = MaxPooling2D((1, 2), border_mode='same', name='mxp1')(x)
    x = Conv2D(nFilters, int(convSize/2), strides=(1,1), activation='tanh', padding='same',
               name='conv2', activity_regularizer=regularizers.l1(reg))(x)

    x = MaxPooling2D((1, 2), border_mode='same', name='mxp2')(x)
    x = Conv2D(nFilters, int(convSize/4), strides=(1,1), activation='tanh', padding='same',
               name='conv3', activity_regularizer=regularizers.l1(reg))(x)

    x = MaxPooling2D((1, 2), border_mode='same', name='mxp3')(x)
    x = Conv2D(nFilters, int(convSize/8), strides=(1,1), activation='tanh', padding='same',
               name='conv4', activity_regularizer=regularizers.l1(reg))(x)
    #########
    x = Conv2D(nFilters, int(convSize/8), strides=(1,1), activation='tanh', padding='same',
               name='conv5', activity_regularizer=regularizers.l1(reg))(x)

    x = UpSampling2D((1, 2), name='ups1')(x)
    x = Conv2D(nFilters, int(convSize/4), strides=(1,1), activation='tanh', padding='same',
               name='conv6', activity_regularizer=regularizers.l1(reg))(x)

    x = UpSampling2D((1, 2), name='ups2')(x)
    x = Conv2D(nFilters, int(convSize/2), strides=(1,1), activation='tanh', padding='same',
               name='conv7', activity_regularizer=regularizers.l1(reg))(x)

    x = UpSampling2D((1, 2), name='ups3')(x)
    decoded = Conv2D(1, convSize, strides=(1,1), activation='tanh', padding='same',
                     name='conv_out', activity_regularizer=regularizers.l1(reg))(x)
                     
    autoencoder = Model(input_img, decoded)

    print("\n\n")
    autoencoder.summary()
    print("\n\n")

    #Novo optimizer, versão 7
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    autoencoder.compile(loss='mean_squared_error', optimizer=sgd, 
                        metrics=['accuracy'])
    #autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    #patience é o nEpochs que passa sem melhora até que o treinamento pare
    earlyStopping=callbacks.EarlyStopping(monitor='val_loss', patience=4, 
                                          verbose=1, mode='auto')
    
    history = autoencoder.fit(X_train, X_test, epochs=nEpochs, 
                              batch_size=batchSize, shuffle=True, 
                              validation_data=(X_test, X_test), verbose=True,
                              callbacks=[earlyStopping])

    #autoencoder.save(modelNameToSave)  # creates a HDF5 file 'my_model.h5'
    autoencoder_json = autoencoder.to_json()
    with open(modelNameToSave, 'w') as f:
        f.write(autoencoder_json)
    # serialize weights to HDF5
    autoencoder.save_weights(weightsNameToSave)
    print("\n Saved model and weights to %s and %s." % (modelNameToSave, 
                                                     weightsNameToSave))

    end=time.time()
    elapsed_time = (end-start)/60    
    print("Time of execution = %s minutes." % (elapsed_time))

    del autoencoder  # deletes the existing model

########################################################
########################################################
########################################################
def evalModel(X, Y, modelName, weightsName, batchSize):

    with open(modelName, 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights(weightsName)
    print("\nLoaded model and weights.")

    # evaluate loaded model on test data
#    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop',
#                          metrics=['accuracy'])
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer='sgd', 
                        metrics=['accuracy'])
                        
    score = model.evaluate(X, Y, verbose=0, batch_size=batchSize)

    print("Obtaining predicted model...")
    pred = model.predict(X, batch_size=batchSize)
    
    plt.figure(1)
    plotData(X[0][0][1], 'X', Y[0][0][1], 'Y', pred[0][0][1], 'pred')
    plotData(X[0][0][1], 'X', Y[0][0][1], 'Y', pred[0][0][1], 'pred')
    plotData(X[10][0][0], 'X', Y[10][0][0], 'Y', pred[10][0][0], 'pred')


    print("%s: %.2f%%" % (model.metrics_names[0], score[0]*100)) #loss
    #print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100)) #accuracy
    return

########################################################
########################################################
########################################################

def initNetworkParameters(): 
    nEpochs = 100
    batchSize = 4
    modelName, weightsName = getNetworkNames()
    nChannels = 19
    windW = 7680#8064

    return (nEpochs, batchSize, modelName, weightsName,
            nChannels, windW)

###################################################
###################################################
###################################################
###################################################
###################################################

def main():

    (nEpochs, batchSize, modelNameToSave, 
     weightsNameToSave, nChannels, windW) = initNetworkParameters()
    
    print("modelName: %s \nweightsName: %s" % (modelNameToSave, weightsNameToSave))

    all_files = prepareFilenames() #prepara nomes de todos os files para uso
    max_files = 53 # max. que a memória aguenta sem processar em batch
    files = all_files[0:80] # numero de files a ser usado
    #print(files)
    
    data = loadData(files)
    ######
    seed = 1 #semente para embaralhar os videos

    qdeBlink = 0.85 # percent of blink to add (max = 1)
    ampBlink = 0.95 # amplitude of blink to add (max = 1)
    subjNoise = 'C.EGC' # person to take blink from

    data, data_noise, ordem_vids_shuf = process_and_addBlink(data, seed, qdeBlink,
                                                             ampBlink, subjNoise, files)
    ######

    (X_train, X_train_noise, X_test,  
     X_test_noise, X_dev, X_dev_noise) = separateSets(data, data_noise, files)
    ######
    '''
    vid = 0
    
    plt.figure(1)
    for k in range(1,11):
        plt.subplot(5,2,k)
        plt.plot(X_train[vid][k-1])
        plt.title(k-1)
    plt.show()

    plt.figure(2)
    for k in range(1,10):
        plt.subplot(5,2,k)
        plt.plot(X_train[vid][k-1+10])
        plt.title(k-1+10)
    plt.show()
    del vid
    '''
    '''
    train, test, dev = separateSets2()    
    
    '''


###################################################################################
    print("Reshaping data...")
    X_train = reshapeData(X_train, nChannels, windW)
    X_train_noise = reshapeData(X_train_noise, nChannels, windW)
    
    X_test = reshapeData(X_test, nChannels, windW)
    X_test_noise = reshapeData(X_test_noise, nChannels, windW)
    
    X_dev = reshapeData(X_dev, nChannels, windW)
    X_dev_noise = reshapeData(X_dev_noise, nChannels, windW)
    
    '''
    plt.figure(1), plt.plot(X_train[0][0][1]) 
    plt.figure(2), plt.plot(X_train_noise[0][0][1])
    plt.show()
    '''    
###################################################################################

    #launch extra terminal to monitor GPU process
    #launchGPUMonitor()

    #pdb.set_trace()
    
    #trainConvolutionalAE(X_train_noise, X_train, nEpochs, batchSize, 
    #                         modelNameToSave, weightsNameToSave, nChannels, windW)

    evalModel(X_dev_noise, X_dev, modelNameToSave, weightsNameToSave, batchSize)

    
    return


########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
if __name__ == '__main__':

    main()
################################################################################
    




