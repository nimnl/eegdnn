# -*- coding: utf-8 -*-
'''
uso: ??
'''
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import pdb


def getLosses(filename):
    '''recebe arquivo no formato .txt, contendo o log (history) do treinamento.
       pegará somente os valores do loss, conforme evoluem com o tempo, e 
       coloca em um vetor separado.
    '''
    losses = open(filename, 'r').read().splitlines()
    lossarray = np.zeros((len(losses)))
    k = 0
    for line in range(0, len(losses)):
        if ' loss' in losses[line]:
            lossarray[k] = float(losses[line].split('loss: ')[1].split(' -')[0])
            k += 1
    return lossarray, losses


def getLosses_macroEpochs(filename):
    '''recebe arquivo no formato .txt, contendo o log (history) do treinamento.
       pegará somente os valores do loss, conforme evoluem com o tempo, e 
       coloca em uma matrix cuja primeira dimensão é a macroépoca e a segunda,
       seus losses correspondentes, sequencialmente.
    '''
    totalmacroepochs = 100 # total de macroepochs
    losses_per_macroepoch = int((1320/4)*9)
    losses_txt = open(filename, 'r').read().splitlines()
    lossarray = np.zeros((totalmacroepochs,2640))#losses_per_macroepoch))#283470))
    k = 0
    macroepoch = 0

    for line in range(0, len(losses_txt)): #para cada linha no arquivo txt
        
        if 'fileBatchNumber: 2 | macroepoch' in losses_txt[line]:
        # only counts a macroepoch when next 9 batches start (at number 2)
            macroepoch = int(losses_txt[line].split('macroepoch: ')[1]) - 1 
            k = 0
            # reason for '-1': indexing starts at 0 but macroepoch counting starts at 1)
            #print('macroepoch:%s' % macroepoch)
            #print('line:%s' % line)

        if ' loss' in losses_txt[line]:
            lossarray[macroepoch][k] = float(losses_txt[line].split('loss: ')[1].split(' -')[0])
            k += 1
            

    return lossarray, losses_txt

def plotLosses(lossarray, macroEpochToPlot, figno):

    loss_min = np.min(lossarray[np.nonzero(lossarray)])
    loss_max = np.max(lossarray[np.nonzero(lossarray)])

    nepochs = np.linspace(1,9, num=np.shape(lossarray)[1])
    plt.figure(figno)
    plt.plot(nepochs, lossarray[macroEpochToPlot][:]) #plota os losses para a macroepoca escolhida
    plt.title('Macroépoca %d' % int(macroEpochToPlot+1), fontsize=16)
    plt.xlabel('No. de batches', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.ylim(loss_min, loss_max)

    plt.show()

    return

if __name__ == "__main__":
    
    filename = 'log30_07_18_COMPLETE.txt'
#    filename = 'log20_08_18.txt'
    lossarray, losses_txt = getLosses_macroEpochs(filename)
    
    #print(*lossarray[0][:20]) # printa até o 20o elemento da 1a macroepoca do lossarray
    '''
    concatenated = np.array([0])#np.zeros((np.shape(lossarray)[0]*np.shape(lossarray)[1]))
    for macroepoch in range(0, np.shape(lossarray)[0]):
        concatenated = np.concatenate((concatenated, lossarray[macroepoch]), axis=None) 
    '''
    #plotLosses(concatenated, macroEpochToPlot=1, figno=1)
    macroEpochToPlot = int(sys.argv[1])
    plotLosses(lossarray, macroEpochToPlot, figno=2)


    
    

