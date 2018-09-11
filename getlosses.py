# -*- coding: utf-8 -*-
'''
uso: python3 plottxt.py [txt filepath]
'''
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import pdb

def refatCorte(i):
    return list(map(float, i.replace("\n", "").split(",")))

def getLossHistory(filename):
    losses = open(filename, 'r').read().splitlines() # lê cada linha
    corte = np.array(corte)
    corte = np.transpose(corte) # formata em # de canais x # de pontos

    corte = corte[2:-1][:] # remove a 1a linha (Time), a 2a e a última (zeros)

    return corte


if __name__ == "__main__":
    cabecalho = []

    filename = sys.argv[1]

    corte = formataCorte(filename)

    print("\nFilename and path: %s\nDimensions: %s\n" % (filename, np.shape(corte)))

    for line in corte:
        print(*line[:20]) # printa até o 20o elemento de cada canal

    plt.plot(corte[2]) #fp1
    plt.show()
    

