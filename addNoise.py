# -*- coding: utf-8 -*-
'''
uso: python3 addNoise.py [file_saudavel DEAP path] [file_ruido AQUIS_LPC path]


'''
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pickle 
import pdb, os, sys


def load_DAT(filename):
    # Check Python version
    if(sys.version_info[0] == 2): # Python2.x
        import cPickle
        data = cPickle.load(open(filename, 'rb'))
    elif(sys.version_info[0] == 3): # Python3.x
        import pickle
        data = pickle.load(open(filename, 'rb'), encoding='latin1')
    try: # this should contemplate cases where data has attr 'data'
        data = data['data']
    except IndexError:
        data = data
    dims = np.shape(data) #[number of videos, number of channels, data points]
    return data, dims
    
def get_dims(data):
    return np.shape(data)
    
def open_directories(txt):
    dirs = open(txt, 'r').read().splitlines() # cria lista com cada linha
    for k in range(len(dirs)):
        if dirs[k].startswith('dir_data '):
            dir_data = dirs[k][12:-1] # pega só o caminho, sem as aspas simples
        elif dirs[k].startswith('dir_save '):
            dir_save = dirs[k][12:-1]
        elif dirs[k].startswith('dir_data_ruidos'):
            dir_data_ruidos = dirs[k][19:-1]
        elif dirs[k].startswith('dir_save_ruidos'):
            dir_save_ruidos = dirs[k][19:-1]
       # else:
       #     print("Erro na formatação do arquivo de diretórios.\n")
    return dir_data, dir_save, dir_data_ruidos, dir_save_ruidos
    
def refatCorte(i):
    return list(map(float, i.replace("\n", "").split(",")))
    
def load_TXT(filename): #carrega e formata para 23 canais
    corte = open(filename)    
    corte = corte.readlines() # lê cada linha
    corte = [refatCorte(i) for i in corte]
    corte = np.array(corte)
    corte = np.transpose(corte) # formata em (# de canais x # de pontos)
    
    corte, dims_corte = formataEEG(corte, 'corte')

    return corte, dims_corte
    
def formataEEG(vetor, tipo):

    if tipo == 'corte':
    
        vetor = vetor[2:-1] # remove a 1a linha (Time), a 2a, e a última (zeros)
        #corte = preencheCorte(corte, num_chan=40) # o eeg saudável DEAP tem 40 canais
        dims_corte = get_dims(vetor)
        
        return vetor, dims_corte
        
    elif tipo == 'deap': 
        FREQSAMPL = 128
        tempo = 3 # segundos

        vetor = vetor[:, :, FREQSAMPL*tempo:] #remove 3 primeiros segundos
        vetor = remapDEAP(vetor)
        return vetor
    
        
def preencheCorte(vetor, num_chan):
# preenche com canais-fantasma, cheios de zero, para igualar dimensões
# com o eeg vindo da DEAP
    [chans, points] = get_dims(vetor)
    
    for k in range(num_chan - chans): #preenche vetor até que num_chan == chans
        vetor = np.append(vetor, [np.zeros(points)], axis=0) #append linhas zeradas
    
    del chans, points
    return vetor
    
def downsampleCorte(corte, fstart, fend):
# reamostra o corte com downsampling para ter mesma frequência de amostragem 
# do eeg_normal, descendo de fstart (ex.: 200Hz) para fend (ex.: 128Hz)

    [chans, points] = get_dims(corte)
    points_final = int(points*(fend/float(fstart)))
    corte_temp = np.zeros((chans, points_final))
    for chan in range(chans):
        corte_temp[chan] = signal.resample(corte[chan], points_final)

    corte = corte_temp
    del points_final, chans, points, corte_temp
    return corte
    
def remapDEAP(vetor):
# vetor de entrada deve ter 40 canais, e é remapeado para no_channels_end
# canais, conforme a organização das aquisições do LPC. 
# >> A informação de canais não usados será perdida. <<
    no_channels_end = 23
    [vids, chans, points] = np.shape(vetor)
    temp = np.zeros((vids, no_channels_end, points))

    if (chans == no_channels_end):
        return vetor #Supõe-se que já está remapeado
    else:
        for vid in range(vids):
            temp[vid][0][:] = vetor[vid][0][:] #Fp1
            temp[vid][1][:] = vetor[vid][16][:] #Fp2
            temp[vid][2][:] = vetor[vid][2][:] # F3
            temp[vid][3][:] = vetor[vid][19][:] # F4
            temp[vid][4][:] = vetor[vid][6][:] # C3
            temp[vid][5][:] = vetor[vid][24][:] # C4
            temp[vid][6][:] = vetor[vid][10][:] # P3
            temp[vid][7][:] = vetor[vid][28][:] # P4
            temp[vid][8][:] = vetor[vid][13][:] # O1
            temp[vid][9][:] = vetor[vid][31][:] # O2
            temp[vid][10][:] = vetor[vid][3][:] # F7
            temp[vid][11][:] = vetor[vid][20][:] # F8
            temp[vid][12][:] = vetor[vid][7][:] # T3 <- T7 (nova nomenclatura)
            temp[vid][13][:] = vetor[vid][25][:] # T4 <- T8 (idem)
            temp[vid][14][:] = vetor[vid][11][:] # T5 <- P7 (idem)
            temp[vid][15][:] = vetor[vid][29][:] # T6 <- P8 (idem)
            temp[vid][16][:] = temp[vid][16][:]  # A1 não existe em DEAP. Ficou com zeros.
            temp[vid][17][:] = temp[vid][17][:] # A2 não existe em DEAP. Ficou com zeros.
            temp[vid][18][:] = vetor[vid][18][:] # Fz
            temp[vid][19][:] = vetor[vid][23][:] # Cz
            temp[vid][20][:] = vetor[vid][15][:] # Pz
            temp[vid][21][:] = vetor[vid][32][:] # X2-X3 <- horizontalEOG
            temp[vid][22][:] = vetor[vid][33][:] # X1 <- verticalEOG

        vetor = temp
        del temp, vids, chans, points
        return vetor


def padCorte(corte, len_final):
# faz zero padding no corte, centralizando-o e fazendo com que suas
# laterais se extendam com zeros até atingirem o tamanho em len_final
    chans, points = np.shape(corte)
    zero_pad_len = len_final - points # tamanho do zero padding

    if (zero_pad_len <= 0): # significa que tamanho(corte) > tamanho(eeg)
        corte = corte[:,0:zero_pad_len] #limita corte a tamanho(eeg)
        return corte
    else:
        zero_pad = np.zeros((int(zero_pad_len/2)))

        corte_pad = np.zeros((chans, len_final))
    
        for chan in range(chans):
            corte_pad[chan] = np.concatenate((zero_pad, corte[chan], zero_pad, 
                                              zero_pad))[:][0:len_final]
        #para evitar erros por conta de poucos pontos em falta, adicionei o padding
        #com uma margem extra de zero_pad pontos (além da margem de zero_pad pontos
        # à direita) e aí trunquei no tamanho desejado para o corte (len_final)

        return corte_pad


def somaRuido(eeg, ruido, quantidade, amplitude):
    # quantidade é a porcentagem de ruído a ser add: 0.2 = 20%
    # amplitude é o fator a multiplicar a amplitude do ruído 
    [chans_ruido, pts_ruido] = np.shape(ruido)
    [vids_eeg, chans_eeg, pts_eeg] = np.shape(eeg)
        
    start_qde = int(pts_ruido*quantidade)
    
    if (quantidade < 0) or (quantidade > 1):
        print("Erro: A quantidade de ruído deve ser uma porcentagem, número entre 0 e 1.")
        return
    elif chans_eeg != chans_ruido:
        print("Erro: O ruído e o EEG saudável devem ter a mesma quantidade de canais.")
        return

    else:
        ruido = ruido[:, :start_qde] # pega os "quantidade %" iniciais do ruído
        ruido = ruido*amplitude #aplica um fator de escala ao ruido
        ruido = padCorte(ruido, pts_eeg) #zero-pad para ruído ter mesmo tamanho de EEG
        temp = np.zeros((vids_eeg, chans_eeg, pts_eeg))

        for vid in range(vids_eeg):
            for chan in range(chans_eeg):
                temp[vid][chan] = eeg[vid][chan] + ruido[chan]
    
    eeg_ruidoso = temp

    del temp
    del chans_ruido, pts_ruido, vids_eeg, chans_eeg, pts_eeg
    return eeg_ruidoso



if __name__ == "__main__":
    
    filename_eeg = sys.argv[1] # caminho para o arquivo saudável
    filename_corte = sys.argv[2] # caminho para o arquivo de corte
    dirs_file = 'dirs.txt'
    
    [dir_data, dir_save] = open_directories(dirs_file)
    os.chdir(dir_data)
    
    print("Lendo arquivo saudável e corte de ruído...")
    [eeg_normal, dims] = load_DAT(filename_eeg)
    [corte, dims_corte] = load_TXT(filename_corte)
    
    print("O arquivo saudável tem dimensões: %s" % (dims,))
    print("O arquivo de corte de ruído a 200Hz tem dimensões: %s" % (dims_corte,))

    corte = downsampleCorte(corte, 200, 128)    

    print("O corte de ruído a 128Hz agora tem dimensões: %s" % (get_dims(corte),))
    print("Formatando EEG normal para 23 canais e eliminando os 3 segundos iniciais...")
    eeg_normal_format = formataEEG(eeg_normal, 'deap')
    print("O arquivo saudável formatado tem dimensões: %s" % (get_dims(eeg_normal_format),))

    eeg_ruidoso = somaRuido(eeg_normal_format, corte, 0.2, 0.5)

    plt.figure(1)
    plt.plot(eeg_normal_format[0][0])
    plt.title('EEG original')
    plt.ylim(-90, 90)

    plt.figure(2)
    plt.plot(eeg_ruidoso[0][0])
    plt.title('EEG com ruido adicionado')
    plt.ylim(-90, 90)

    plt.show()
    

