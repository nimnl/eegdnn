# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import math, time

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

#import cPickle
import pdb # para debugar
import networkx as NX
#from pyhdf import SD, SDC

from scipy import io
#from ffnet import ffnet, mlgraph, savenet, loadnet, exportnet
from scipy.signal import butter, lfilter
from scipy.signal import freqz

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

#TF_CPP_MIN_LOG_LEVEL=2 #filter out warnings regarding memory usage


#####################################################
#####################################################
#####################################################
def trainConvolutionalAE(X_train, X_test, nEpochs, batchSize, modelNameToSave, weightNameToSave, nChannels, windW):
    print("Getting start time for training...")
    
    start=time.time()

    print ("Training Conv. AE...")
    

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Set input dimensions:
    input_img = Input(shape=(1, nChannels, windW))

    convSize = 8
    nFilters = 100

    print ("batchSize: %d | nFilters: %d | %d subjects \n" % (batchSize, nFilters, 
                                                           X_train.shape[0]/40))
    
    x = Conv2D(nFilters, convSize, strides=(1,1), activation='tanh', padding='same',
               name='conv1', activity_regularizer=regularizers.l1(0.01))(input_img)
    x = MaxPooling2D((1, 2), border_mode='same', name='mxp1')(x)
    
    x = Conv2D(nFilters, int(convSize/2), strides=(1,1), activation='tanh', padding='same',
               name='conv2', activity_regularizer=regularizers.l1(0.01))(input_img)
    x = MaxPooling2D((1, 2), border_mode='same', name='mxp2')(x)

    x = Conv2D(nFilters, int(convSize/4), strides=(1,1), activation='tanh', padding='same',
               name='conv3', activity_regularizer=regularizers.l1(0.01))(x)
    x = UpSampling2D((1, 2), name='ups1')(x)

    x = Conv2D(nFilters, int(convSize/2), strides=(1,1), activation='tanh', padding='same',
               name='conv4', activity_regularizer=regularizers.l1(0.01))(x)
    '''    
    x = Dropout(rate=0.2)(x)
    x = Dropout(rate=0.2)(x)
    '''
    decoded = Conv2D(1, convSize, strides=(1,1), activation='tanh', padding='same',
                     name='conv5', activity_regularizer=regularizers.l1(0.01))(x)
                     
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
    earlyStopping=callbacks.EarlyStopping(monitor='val_loss', patience=10, 
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
    autoencoder.save_weights(weightNameToSave)
    print("Saved model and weights.")


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
    print("Loaded model and weights.")

    # evaluate loaded model on test data
#    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop',
#                          metrics=['accuracy'])
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer='sgd', 
                        metrics=['accuracy'])
                        
    score = model.evaluate(X, Y, verbose=0, batch_size=batchSize)
    print("%s: %.2f%%" % (model.metrics_names[0], score[0]*100))
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    return

def reshapeData(data, nChannels, windW):
    data = data.reshape(data.shape[0], 1, nChannels, windW)
    return data


def remapDEAP2(vetor):
# vetor de entrada deve ter 23 canais, e os canais A1, A2 e X1, X2, X3 são 
# excluidos para não interferirem nos treinamentos e validações
    no_channels_end = 19
    [vids, chans, points] = np.shape(vetor)
    temp = np.zeros((vids, no_channels_end, points))
    for vid in range(vids):
#        for chan in range(0, 16):#de 0 a 15
#            temp[vid][chan][:] = vetor[vid][chan][:]
        temp[vid][0][:] = vetor[vid][0][:] 
        temp[vid][1][:] = vetor[vid][1][:] 
        temp[vid][2][:] = vetor[vid][2][:] 
        temp[vid][3][:] = vetor[vid][3][:] 
        temp[vid][4][:] = vetor[vid][4][:] 
        temp[vid][5][:] = vetor[vid][5][:] 
        temp[vid][6][:] = vetor[vid][6][:] 
        temp[vid][7][:] = vetor[vid][7][:] 
        temp[vid][8][:] = vetor[vid][8][:] 
        temp[vid][9][:] = vetor[vid][9][:] 
        temp[vid][10][:] = vetor[vid][10][:] 
        temp[vid][11][:] = vetor[vid][11][:]
        temp[vid][12][:] = vetor[vid][12][:] 
        temp[vid][13][:] = vetor[vid][13][:] 
        temp[vid][14][:] = vetor[vid][14][:] 
        temp[vid][15][:] = vetor[vid][15][:] 
       
        temp[vid][16][:] = vetor[vid][18][:] # Fz
        temp[vid][17][:] = vetor[vid][19][:] # Cz
        temp[vid][18][:] = vetor[vid][20][:] # Pz
    del vetor
    vetor = temp
    del temp, vids, chans, points
    return vetor

def launchGPUMonitor():
    os.system('gnome-terminal -x watch -n 1 nvidia-smi')
    return


########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
if __name__ == '__main__':

################################################################################
    files =  ['s01.dat', 's02.dat', 's03.dat', 's04.dat', 's05.dat', 's06.dat', 
               's07.dat', 's08.dat', 's09.dat', 's10.dat', 's11.dat', 's12.dat', 
               's13.dat', 's14.dat', 's15.dat', 's16.dat', 's17.dat', 's18.dat', 
               's19.dat', 's20.dat', 's21.dat', 's22.dat', 's23.dat', 's24.dat', 
               's25.dat', 's26.dat', 's27.dat', 's28.dat', 's29.dat', 's30.dat',
               's31.dat', 's32.dat']
    files = files[0:10] # usa até o s04.dat

################################################################################
    data_dims = np.shape(preprocess.load_file(files[0])[0]) #shape de uma pessoa
    
    #concatenaremos todos os participantes em um único vetor data:
    data_dims = (len(files)*data_dims[0], data_dims[1], data_dims[2])
    data = np.zeros(data_dims)

    print("Loading files...")
    for file in range(len(files)):
        data[file*40:(file+1)*40] = preprocess.load_file(files[file])[0]
##################################################################################
    seed = 1 #semente para embaralhar os videos
    
    data, ordem_vids_shuf = preprocess.preprocessData(data, seed)

    quantidade_blink = 0.85
    amplitude_blink = 0.95
    subject_noise = 'C.EGC'
    
    data_noise = np.zeros(np.shape(data))
    for file in range(len(files)):
        data_noise[file*40:(file+1)*40] = preprocess.addBlink(data[file*40:(file+1)*40], 
                                                              quantidade_blink, 
                                                              amplitude_blink, 
                                                              subject_noise)
    # Remapear de 23 para 19 canais (usar somente os de EEG):
    no_channels_end = 19
    temp1 = np.zeros((np.shape(data)[0], no_channels_end, np.shape(data)[2]))
    temp2 = np.zeros((np.shape(data)[0], no_channels_end, np.shape(data)[2]))

    for file in range(len(files)):
        temp1[file*40:(file+1)*40] = remapDEAP2(data[file*40:(file+1)*40])
    data = temp1
    
    for file in range(len(files)):
        temp2[file*40:(file+1)*40] = remapDEAP2(data_noise[file*40:(file+1)*40])
    data_noise = temp2


###################################################################################

    X_train = data[0:(len(files)-2)*40] #30 primeiras pessoas (1200 primeiros vídeos)
    X_dev = data[(len(files)-2)*40:(len(files)-1)*40] #31a pessoa
    X_test = data[(len(files)-1)*40:(len(files))*40] #32a pessoa
    
    X_train_noise = data_noise[0:(len(files)-2)*40] #30 primeiras pessoas (1200 primeiros vídeos)
    X_dev_noise = data_noise[(len(files)-2)*40:(len(files)-1)*40] #31a pessoa
    X_test_noise = data_noise[(len(files)-1)*40:(len(files))*40] #32a pessoa
###################################################################################
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
    #exclude certain people from training and put them in validation
    train = np.zeros((len(a), np.shape(a)[0]-1))
    test = dev = np.zeros((len(a)))
    for which_exclude in range(0, len(a)):
        train[which_exclude][:] = a[a!=a[((which_exclude+1)%(len(a)+1))]]
        test[which_exclude] = a[a==a[(which_exclude)]]
        dev[which_exclude] = a[a==a[((which_exclude+1)%(len(a)+1))]]
    '''

###################################################################################
    nEpochs = 50
    batchSize = 4
    modelNameToSave = 'deepNetwork1.json'
    weightsNameToSave = 'deepWeights1.h5'
    nChannels = 19
    windW = 7680#8064
###################################################################################
    print("Reshaping data...")
    X_train = reshapeData(X_train, nChannels, windW)
    X_train_noise = reshapeData(X_train_noise, nChannels, windW)
    
    X_test = reshapeData(X_test, nChannels, windW)
    X_test_noise = reshapeData(X_test_noise, nChannels, windW)
    
    X_dev = reshapeData(X_dev, nChannels, windW)
    X_dev_noise = reshapeData(X_dev_noise, nChannels, windW)

    plt.figure(1), plt.plot(X_train[0][0][1]) 
    plt.figure(2), plt.plot(X_train_noise[0][0][1])
    plt.show()

###################################################################################

    #launch extra terminal to monitor GPU process
    #launchGPUMonitor()

    trainConvolutionalAE(X_train_noise, X_train, nEpochs, batchSize, 
                         modelNameToSave, weightsNameToSave, nChannels, windW)
         

    evalModel(X_dev_noise, X_dev, modelNameToSave, weightsNameToSave, batchSize)




























