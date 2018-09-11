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
#import networkx as NX

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

def getXCorrelation(comRuido, semRuido, pred):
#Teremos um numero de XCorr para cada canal do conjunto, comparando
#o conjunto predito com o conjunto com ruído e sem ruído através
#dos coeficientes da matrix de correlação
    xcorr_comRuido = np.zeros((np.shape(comRuido)[0], np.shape(comRuido)[2])) #(vids, chans) from [vids][1][chans][points]
    xcorr_semRuido = np.zeros((np.shape(comRuido)[0], np.shape(comRuido)[2])) 

    for vid in range(0, np.shape(comRuido)[0]):
        for chan in range(0, np.shape(comRuido)[2]):
            # np.corrcoef gives 2x2 matrix. We then take either a12 or a21:
            xcorr_comRuido[vid][chan] = np.corrcoef(comRuido[vid][0][chan], pred[vid][0][chan])[0][1] #a12
            xcorr_semRuido[vid][chan] = np.corrcoef(semRuido[vid][0][chan], pred[vid][0][chan])[0][1] #a12
        
    return xcorr_comRuido, xcorr_semRuido


def getStats(input2D_semRuido, input2D_comRuido):
#computes mean and stdev across desired axis for a 2D-matrix.
#axis=0 -> across each row. axis=1 -> across each column 
#we are interested in the mean and stdev for each channel, across all videos -> axis=0
#semRuido and comRuido are not to be followed necessarily. It's just for reference and
#a legacy naming scheme from the first time I created this function

    means_semRuido = np.zeros((np.shape(input2D_semRuido)[1])) # no. of channels
    stdevs_semRuido = np.zeros((np.shape(input2D_semRuido)[1]))
    
    means_comRuido = np.zeros((np.shape(input2D_comRuido)[1])) 
    stdevs_comRuido = np.zeros((np.shape(input2D_comRuido)[1]))
    
    for vid in range(0, np.shape(input2D_semRuido)[0]): #1st dimension: videos
        for chan in range(0, np.shape(input2D_semRuido)[1]): #2nd dimension: channels
            means_semRuido = np.mean(input2D_semRuido, axis=0)
            stdevs_semRuido = np.std(input2D_semRuido, axis=0)

    for vid in range(0, np.shape(input2D_comRuido)[0]): #1st dimension: videos
        for chan in range(0, np.shape(input2D_comRuido)[1]): #2nd dimension: channels
            means_comRuido = np.mean(input2D_comRuido, axis=0)
            stdevs_comRuido = np.std(input2D_comRuido, axis=0)
            
    return (means_semRuido, stdevs_semRuido,
            means_comRuido, stdevs_comRuido)


def getPSNR(input_I, input_K):
#inputs I and K have dimensions: (vids, 1, chans, points)
#input_I is noiseless, input_K is noisy
    mse = np.zeros((np.shape(input_I)[0], np.shape(input_I)[2])) # (vids, chans)
    PSNR = np.zeros((np.shape(input_I)[0], np.shape(input_I)[2])) 
    
    for vid in range(0, np.shape(mse)[0]):
        for chan in range(0, np.shape(mse)[1]): 
            mse[vid][chan] = np.mean((input_I[vid][0][chan] - input_K[vid][0][chan]) ** 2)

    for vid in range(0, np.shape(PSNR)[0]):
        for chan in range(0, np.shape(PSNR)[1]): 
            if mse[vid][chan] == 0:
                PSNR[vid][chan] = 100
            MAX_I = np.max(input_I[vid][0][chan])
            PSNR[vid][chan] = 20 * np.log10(MAX_I / np.sqrt(mse[vid][chan]))

    return PSNR

def filterData(inputData):
    buttorder=9   
    fs = 128.0
    lowcut = 4.0
    highcut = 45.0

    dataFiltered = np.zeros(np.shape(inputData))

    def butter_bandpass(lowcut, highcut, fs, order=buttorder):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=buttorder):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
 
    for vid in range(0, np.shape(inputData)[0]):
        for chan in  range(0, np.shape(inputData)[2]):
            dataFiltered[vid][0][chan] = butter_bandpass_filter(inputData[vid][0][chan], lowcut, highcut, fs, order=buttorder)

    return dataFiltered


#####################################################
def continueTrainConvolutionalAE(X_train, X_test, X_dev1, X_dev2, nEpochs, batchSize, modelNameToLoad, weightsNameToLoad, modelNameToSave, weightsNameToSave, nChannels, windW):
    print("Getting start time for training...")
    
    start=time.time()
################loading preexisting model
    with open(modelNameToLoad, 'r') as f:
        autoencoder = model_from_json(f.read())

    # Load weights into the new model
    autoencoder.load_weights(weightsNameToLoad)
    print("\nLoaded model and weights from %s and %s." % (modelNameToLoad, 
                                                          weightsNameToLoad))
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
#X is noisy and Y is without noise.

    with open(modelName, 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights(weightsName)
    print("\nPreparing evaluation for test data...| Dimensions: ", X.shape)
    print("\nLoaded model and weights from %s and %s." % (modelName, 
                                                          weightsName))

    # evaluate loaded model on test data
#    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop',
#                          metrics=['accuracy'])
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer='sgd', 
                        metrics=['accuracy'])
                        
    score = model.evaluate(X, Y, verbose=0, batch_size=batchSize)

    print("Obtaining predicted model...")
    pred = model.predict(X, batch_size=batchSize)

    print("Filtering noisy data with Butterworth traditional filter...")
    filtered = filterData(X)


    plt.figure(1)
    plotData(X[4][0][1], 'X (com ruído)', Y[4][0][1], 'Y (sem ruído)', pred[4][0][1], 'pred')
    plt.figure(2)
    plotData(X[4][0][1], 'X (com ruído)', Y[4][0][1], 'Y (sem ruído)', filtered[4][0][1], 'filtered (traditionally)')
#    plotData(X[8][0][5], 'X (com ruído)', Y[8][0][5], 'Y (sem ruído)', pred[8][0][5], 'pred')
#    plotData(X[10][0][0], 'X (com ruído)', Y[10][0][0], 'Y (sem ruído)', pred[10][0][0], 'pred')
##  plotData(X[vid][0][chan], 'X (com ruído)', Y[vid][0][chan], 'Y (sem ruído)', pred[vid][0][chan], 'pred')
    plt.show()

    [xcorr_comRuido, xcorr_semRuido] = getXCorrelation(X, Y, pred)
    [xcorr_comRuido_filtered, xcorr_semRuido_filtered] = getXCorrelation(X, Y, filtered)

    (means_semRuido, stdevs_semRuido,
     means_comRuido, stdevs_comRuido) = getStats(xcorr_semRuido, xcorr_comRuido)

    (means_semRuido_filtered, stdevs_semRuido_filtered,
     means_comRuido_filtered, stdevs_comRuido_filtered) = getStats(xcorr_semRuido_filtered,
                                                                   xcorr_comRuido_filtered)


#     plt.hist(X[20][0][10], bins=100), plt.show() #histogram for a given channel
    print("Generated means and stdevs for the XCorrelation coefficients, per channel.")
    
    PSNR_comRuido = getPSNR(Y, X) #Y: sem ruido, X: com ruido
    PSNR_pred = getPSNR(Y, pred)
    PSNR_filtered = getPSNR(Y, filtered)
        
    (means_PSNR_pred, stdevs_PSNR_pred, 
     means_PSNR_comRuido, stdevs_PSNR_comRuido) = getStats(PSNR_pred, PSNR_comRuido)
#    del stdevs_PSNR_pred, stdevs_PSNR_comRuido
    (means_PSNR_filtered, stdevs_PSNR_filtered, 
     means_PSNR_comRuido, stdevs_PSNR_comRuido) = getStats(PSNR_filtered, PSNR_comRuido)

     #PSNR_ComRuido is being repeated here but I'm lazy to alter the function to receive
     #only one argument now
    
    print("Generated means of PSNRs, per channel.")
        
    PSNR_pred_melhor_comRuido = 0
    PSNR_pred_melhor_filtered = 0
    PSNR_filtered_melhor_comRuido = 0
    PSNR_filtered_melhor_pred = 0
    PSNR_comRuido_melhor_pred = 0
    
    for vid in range(0, np.shape(PSNR_pred)[0]): #1st dimension: videos
            for chan in range(0, np.shape(PSNR_pred)[1]): #2nd dimension: channels
                if PSNR_pred[vid][chan] > PSNR_comRuido[vid][chan]:
                        PSNR_pred_melhor_comRuido += 1
                if PSNR_pred[vid][chan] > PSNR_filtered[vid][chan]:
                        PSNR_pred_melhor_filtered += 1
                if PSNR_filtered[vid][chan] > PSNR_comRuido[vid][chan]:
                        PSNR_filtered_melhor_comRuido += 1
                if PSNR_filtered[vid][chan] > PSNR_pred[vid][chan]:
                        PSNR_filtered_melhor_pred += 1
                if PSNR_comRuido[vid][chan] > PSNR_pred[vid][chan]:
                        PSNR_comRuido_melhor_pred += 1
    pdb.set_trace()

    print("%s: %.2f%%" % (model.metrics_names[0], score[0]*100)) #loss
    #print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100)) #accuracy
    return

########################################################
########################################################
########################################################
def mainBatch(macroepoch, first, train, evaluate):
#This function trains in batches: <filesperBatch> files will be trained for 1 epoch, per macro-epoch
#first is a flag for when this is running for a new (first) model
#train and evaluate are flags for when each case is desired.

    (nEpochs, batchSize, nChannels, windW) = initNetworkParameters()
    filesperBatch = 35 #number of files per batch of files
    fileBatches = 10 #number of fileBatches per macroepoch
    nMacroEpochs = 100
    batchStartFrom = 2 # de onde recomeçar o carregamento da rede no 'for macroepoch'. default = 2
    
    evaluateNumber = 32; #number of files to evaluate


    if train==1:
        if first == 1:
            first = 0

            (X_train, X_train_noise, X_test,  
             X_test_noise, X_dev, X_dev_noise) = prepareALL(0,filesperBatch) #first run
            # always check if nEpochs is being set to 1 (one)!
            
            modelNameToSave, weightsNameToSave = getNetworkNames(fileBatchNumber=1)        

            print("modelName: %s \nweightsName: %s" % (modelNameToSave, weightsNameToSave))
#            pdb.set_trace()
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

    if evaluate == 1: #evaluate the modelName
        (X_train, X_train_noise, X_test,  
         X_test_noise, X_dev, X_dev_noise) = prepareALL(0, evaluateNumber) #prepare <evaluateNumber> files

        #This way we take the last model and weights trained and load them to evaluation:
        modelNameToEval, weightsNameToEval = getNetworkNames(fileBatchNumber=fileBatches-1) #<fileBatches-1> is the last batch
        evalModel(X_test_noise, X_test, modelNameToEval, weightsNameToEval, batchSize) 
    return

###################################################
###################################################
###################################################
def initNetworkParameters(): 
    nEpochs = 1 #due to batch training, one epoch per batch of files
    batchSize = 4 #this is batch of samples (step of progress,during training), not of files 
    #dimensions of the data:
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
    noiseType = 'morder'#'blink'

    data, data_noise, ordem_vids_shuf = process_and_addBlink(data, seed, qdeBlink,
                                                             ampBlink, subjNoise, 
                                                             files, noiseType)
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

    evalModel(X_test_noise, X_test, modelNameToSave, weightsNameToSave, batchSize)

    
    return


########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
if __name__ == '__main__':
    #This script shall be called via a shell script (*.sh)
    #sys.argv[1] e [2] estao definidos em nn_utils.py
    macroEpoch = sys.argv[3]
    first = int(sys.argv[4])
    train = int(sys.argv[5])
    evaluate = int(sys.argv[6])
    
    
    mainBatch(macroEpoch, first, train, evaluate)

     
################################################################################
    




