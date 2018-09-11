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

import preprocess #modulo preprocess.py deve estar na mesma pasta

def plotData(x1, titlex1, x2, titlex2, x3, titlex3):
    plt.subplot(3,1,1)
    plt.plot(x1)
    plt.title(titlex1)
    plt.grid()
    plt.subplot(3,1,2)
    plt.plot(x2)
    plt.title(titlex2)
    plt.grid()
    plt.subplot(3,1,3)
    plt.plot(x3)
    plt.title(titlex3)
    plt.grid()
    #plt.show()
    return

##########################################

def reshapeData(data, nChannels, windW):
    data = data.reshape(data.shape[0], 1, nChannels, windW)
    return data

##########################################

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

##########################################

def launchGPUMonitor():
    os.system('gnome-terminal -x watch -n 1 nvidia-smi')
    return

##########################################

def prepareFilenames():
    files_DEAP =  ['s01.dat', 's02.dat', 's03.dat', 's04.dat', 's05.dat',
                   's06.dat', 's07.dat', 's08.dat', 's09.dat', 's10.dat',
                   's11.dat', 's12.dat', 's13.dat', 's14.dat', 's15.dat',
                   's16.dat', 's17.dat', 's18.dat', 's19.dat', 's20.dat',
                   's21.dat', 's22.dat', 's23.dat', 's24.dat', 's25.dat',
                   's26.dat', 's27.dat', 's28.dat', 's29.dat', 's30.dat',
                   's31.dat', 's32.dat']
 #   dir_all_files = '/$HOME/Documents/PIBITI_2017-2018/DEAP_dataset/ALL_DATA/'
    
    all_sintetico = [ ]
    for sintetico in range(1,11):
        for filename in files_DEAP:
            sintetico_name = ''.join([filename[:-4], '_sint', 
                                      str(sintetico), '.dat'])
            all_sintetico.append(sintetico_name)
    all_files = files_DEAP + all_sintetico
    return all_files

##########################################

def loadData(files):
    '''
    Returns array "data" of all selected people concatenated, in blocks of 40 vids
    eg.: if 4 files were selected ('s01.dat' ... 's04.dat'), the final shape of
    "data" will be (160, 19, 7860), as in 4*40vids, 19 chans/vid and 7860 pts/chan.
    '''
    data_dims = np.shape(preprocess.load_file(files[0])[0]) #shape de uma pessoa

    #concatenaremos todos os participantes em um único vetor data:
    data_dims = (len(files)*data_dims[0], data_dims[1], data_dims[2])
    data = np.zeros(data_dims)

    print("\nLoading files...")
    for file in range(len(files)):
        data[file*40:(file+1)*40] = preprocess.load_file(files[file])[0]
    return data

##########################################

def process_and_addBlink(data, seed, qdeBlink, ampBlink, subjNoise, files, noiseType):
    '''
    - Preprocesses data and shuffles video order per subject (using seed) 
    using the routines in preprocess.preprocessData()
    - adds qdeBlink% of subjNoise's blink with ampBlink amplitude
    - remaps from 23 channels to 19 channels using remapDEAP2()
    - Returns order of shuffled videos, data and data_noise
    - The type of noise (blink, morder, engolir) is a string <'noiseType'>
    that has the same term used in the .txt files with the noise acquisitions
    (corte_blink.txt, corte_morder.txt, corte_engolir.txt ...)
    '''
    # preprocess data:    
    data, ordem_vids_shuf = preprocess.preprocessData(data, seed)
    # add blink to data:
    data_noise = np.zeros(np.shape(data))
    for file in range(len(files)):
        data_noise[file*40:(file+1)*40] = preprocess.addBlink(data[file*40:(file+1)*40], 
                                                              qdeBlink, ampBlink, 
                                                              subjNoise, noiseType)
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

    del temp1, temp2
    return data, data_noise, ordem_vids_shuf

##########################################

def separateSets(data, data_noise, files):
    '''
    Separates training, validation (dev) and test sets for the network.
    Training will have all files minus 2, dev and test will have the other two.
    '''
    X_train = data[0:(len(files)-2)*40] #n-2 primeiras pessoas (40*(n-2) primeiros vídeos)
    X_dev = data[(len(files)-2)*40:(len(files)-1)*40] #(n-1)esima pessoa
    X_test = data[(len(files)-1)*40:(len(files))*40] #(n)esima pessoa
    
    X_train_noise = data_noise[0:(len(files)-2)*40] # idem X_train
    X_dev_noise = data_noise[(len(files)-2)*40:(len(files)-1)*40] #idem X_dev
    X_test_noise = data_noise[(len(files)-1)*40:(len(files))*40] #idem X_test

    return X_train, X_train_noise, X_test, X_test_noise, X_dev, X_dev_noise

##########################################
def separateSets2():
    '''
    NOT FULLY IMPLEMENTED. DO NOT USE.
    Instead of taking the n-2 first people for training and the other 2 for
    dev and test, selects n-2 people from all files always isolating 2 people, 
    not necesarily the last ones. Therefore raising the bar for dev and test.
    '''
    #exclude certain people from training and put them in validation
    train = np.zeros((len(a), np.shape(a)[0]-1))
    test = dev = np.zeros((len(a)))
    for which_exclude in range(0, len(a)):
        train[which_exclude][:] = a[a!=a[((which_exclude+1)%(len(a)+1))]]
        test[which_exclude] = a[a==a[(which_exclude)]]
        dev[which_exclude] = a[a==a[((which_exclude+1)%(len(a)+1))]]

    return []
###########################################
def getNetworkNames(fileBatchNumber):
# use filebatchNumber = 0 if all files are being trained at once
# user another number if 
    errormsg = ("Argument must be '-name' followed by desired filename. "
                 "Or no argument and the name will be chosen automaticaly.")

    if len(sys.argv)>2:
        if sys.argv[1] == '-name':
            name = str(sys.argv[2])
            nameNN = ''.join(['deepNetwork_', name])
            nameW = ''.join(['deepWeights_', name])
        else:
            print(errormsg)
            exit()

    elif len(sys.argv)==2:
        print(errormsg)
        exit()
    
    else:
        date = datetime.datetime.now().strftime("%Y-%m-%d")#_%H:%M") 
        nameNN = ''.join(['deepNetwork_', date])
        nameW = ''.join(['deepWeights_', date])

    if fileBatchNumber != 0: #names de file batch model
        nameNN = '_batch'.join([nameNN, str(fileBatchNumber)])
        nameW = '_batch'.join([nameW, str(fileBatchNumber)])         

    modelName = ''.join([nameNN, '.json'])
    weightsName = ''.join([nameW, '.h5']) 

    return modelName, weightsName




