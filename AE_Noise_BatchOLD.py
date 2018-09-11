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

import gc

import preprocess #modulo preprocess.py deve estar na mesma pasta

# o modulo nn_utils.py abaixo tambem deve estar na mesma pasta!!
from nn_utils import (plotData, reshapeData, remapDEAP2, launchGPUMonitor,
                      prepareFilenames, loadData, process_and_addBlink,
                      separateSets, separateSets2, getNetworkNames)
 
#TF_CPP_MIN_LOG_LEVEL=2 #filter out warnings regarding memory usage

#####################################################
#####################################################
#####################################################
def continueTrainConvolutionalAE(X_train, X_test, X_dev1, X_dev2, nEpochs, batchSize, modelNameToLoad, weightsNameToLoad, modelNameToSave, weightsNameToSave, nChannels, windW):
    print("Getting start time for training...")
    
    start=time.time()
################loading preexisting model
    with open(modelNameToLoad, 'r') as f:
        autoencoder = model_from_json(f.read())

    # Load weights into the new model
    autoencoder.load_weights(weightsNameToLoad)
    print("\nLoaded model and weights.")
########################    
    print ("Training Conv. AE...")
    

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    convSize = 8
    nFilters = 100
    reg = 0

    # Set input dimensions:
    print ("batchSize: %d | nFilters: %d | %d subjects \n" % (batchSize, nFilters, 
                                                              X_train.shape[0]/40))
    
    # Optimizer:
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    autoencoder.compile(loss='mean_squared_error', optimizer=sgd, 
                        metrics=['accuracy'])
    
    #patience é o nEpochs que passa sem melhora até que o treinamento pare
    earlyStopping=callbacks.EarlyStopping(monitor='val_loss', patience=4, 
                                          verbose=1, mode='auto')
    
    history = autoencoder.fit(X_train, X_test, epochs=nEpochs, 
                              batch_size=batchSize, shuffle=True, 
                              validation_data=(X_dev1, X_dev2), verbose=True,
                              callbacks=[earlyStopping])

    #autoencoder.save(modelNameToSave)  # creates a HDF5 file 'my_model.h5'
    autoencoder_json = autoencoder.to_json()
    with open(modelNameToSave, 'w') as f:
        f.write(autoencoder_json)
    # serialize weights to HDF5
    autoencoder.save_weights(weightsNameToSave)
    print("\nSaved model and weights to %s and %s." % (modelNameToSave, 
                                                     weightsNameToSave))

    end=time.time()
    elapsed_time = (end-start)/60    
    print("Time of execution = %s minutes." % (elapsed_time))

    del autoencoder  # deletes the existing model

############################################################
############################################################
############################################################
############################################################
############################################################
############################################################


def trainConvolutionalAE(X_train, X_test, X_dev1, X_dev2, nEpochs, batchSize, modelNameToSave, weightsNameToSave, nChannels, windW):
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
                              validation_data=(X_dev1, X_dev2), verbose=True,
                              callbacks=[earlyStopping])

    #autoencoder.save(modelNameToSave)  # creates a HDF5 file 'my_model.h5'
    autoencoder_json = autoencoder.to_json()
    with open(modelNameToSave, 'w') as f:
        f.write(autoencoder_json)
    # serialize weights to HDF5
    autoencoder.save_weights(weightsNameToSave)
    print("\nSaved model and weights to %s and %s." % (modelNameToSave, 
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
def mainBatch(macroepoch, first):
#Trains in batches of filesperBatch files will be trained for 1 epoch, per macro-epoch
#first is a flag for when this is running for a new (first) model

    (nEpochs, batchSize, nChannels, windW) = initNetworkParameters()
    filesperBatch = 35 #number of files per batch of files
    fileBatches = 10 #number of fileBatches per macroepoch
    nMacroEpochs = 100
    batchStartFrom = 2 # de onde recomeçar o carregamento da rede no 'for macroepoch'

    if first == 1:
        first = 0

        (X_train, X_train_noise, X_test,  
         X_test_noise, X_dev, X_dev_noise) = prepareALL(0,filesperBatch) #first run
        # always check if nEpochs is being set to 1 (one)!
        
        modelNameToSave, weightsNameToSave = getNetworkNames(fileBatchNumber=1)        

        print("modelName: %s \nweightsName: %s" % (modelNameToSave, weightsNameToSave))

        trainConvolutionalAE(X_train_noise, X_train, X_dev_noise, X_dev, nEpochs, batchSize, 
                             modelNameToSave, weightsNameToSave, nChannels, windW)

    #for macroepoch in range(1, nMacroEpochs+1): # same as nEpochs in non-batch training
    for fileBatchNumber in range(batchStartFrom,fileBatches):# 9 batches of 35 .dat files
        print("--------------------------------------------------")            
        print("\n Running for fileBatchNumber: %s | macroepoch: %s" %(fileBatchNumber, macroepoch))
        modelNameToSave, weightsNameToSave = getNetworkNames(fileBatchNumber)
        modelNameToLoad, weightsNameToLoad = getNetworkNames(fileBatchNumber-1)
        (X_train, X_train_noise, X_test, 
         X_test_noise, X_dev, X_dev_noise) = prepareALL(fileBatchNumber*filesperBatch, 
                                                        (fileBatchNumber+1)*filesperBatch)
        continueTrainConvolutionalAE(X_train_noise, X_train, X_dev_noise, X_dev,
                                     nEpochs, batchSize, 
                                     modelNameToLoad, weightsNameToLoad, 
                                     modelNameToSave, weightsNameToSave, 
                                     nChannels, windW)


    return

###################################################
###################################################
###################################################
def initNetworkParameters(): 
    nEpochs = 1
    batchSize = 4
    nChannels = 19
    windW = 7680#8064

    return (nEpochs, batchSize, nChannels, windW)

###################################################
###################################################
###################################################
###################################################
###################################################
###################################################
###################################################
###################################################
###################################################
###################################################

def prepareALL(start, end):
    '''    
    prepares all files from number start to number end.
    example: prepareALL(12,72) prepares from file 12 to file 72.
    '''
    (nEpochs, batchSize, nChannels, windW) = initNetworkParameters()
    modelNameToSave, weightsNameToSave = getNetworkNames(fileBatchNumber=0)
    #fileBatchNumber=0 means all files will be processed and trained at once

    all_files = prepareFilenames() #prepara nomes de todos os files para uso
    files = all_files[start:end] # numero de files a ser usado
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
    return (X_train, X_train_noise, X_test, X_test_noise, 
            X_dev, X_dev_noise)


###################################################
###################################################
###################################################
###################################################
###################################################
###################################################    
###################################################
###################################################
###################################################

def main():

    (nEpochs, batchSize, nChannels, windW) = initNetworkParameters()
    modelNameToSave, weightsNameToSave = getNetworkNames(fileBatchNumber=0)
    
    print("modelName: %s \nweightsName: %s" % (modelNameToSave, weightsNameToSave))
    max_files = 53 # max. que a memória aguenta de uma vez sem processar em batch
    (X_train, X_train_noise, X_test,  
         X_test_noise, X_dev, X_dev_noise) = prepareALL(0, max_files) # first 53 files



    #launch extra terminal to monitor GPU process
    #launchGPUMonitor()

    #pdb.set_trace()
    
    trainConvolutionalAE(X_train_noise, X_train, nEpochs, batchSize, 
                         modelNameToSave, weightsNameToSave, nChannels, windW)

    evalModel(X_dev_noise, X_dev, modelNameToSave, weightsNameToSave, batchSize)

    
    return


########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
if __name__ == '__main__':
    #sys.argv[1] e [2] estao definidos em nn_utils.py
    macroEpoch = sys.argv[3]
    first = int(sys.argv[4])
    mainBatch(macroEpoch, first)
################################################################################
    




