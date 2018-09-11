# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
#import cPickle, pickle # para abrir os arquivos .dat, importados na função
import os, sys, pdb
import random # para a função de aleatorizar ordem dos vídeos
import addNoise # deve estar na mesma pasta

def load_file(filename):
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


def normalizaAmp(vetor): # use for each channel, in a loop for all chans.
    max_abs = max(vetor, key=abs)
    if max_abs == 0:
        return vetor
    vetor_norm = vetor/abs(max_abs)
    return vetor_norm


def shuffleVids(conjunto, seed): # conjunto é o de 40 vídeos por sujeito
    random.seed(seed)
    qde_videos = np.shape(conjunto)[0] # 1a dimensão do array conjunto
    ordem = random.sample(range(qde_videos), qde_videos)
    # o vid-ésimo vídeo de conjunto_shuffled será o vid-ésimo vídeo de conjunto
    # de acordo com ordem. Isto é, se o primeiro valor de ordem for 23, o 
    # vid-ésimo vídeo de conjunto_shuffled será o 24o vídeo de conjunto.
    conjunto = conjunto[ordem] #conjunto agora é conjunto_shuffled
    return conjunto, ordem

def shuffleVidsOLD(conjunto, seed): # conjunto é o de 40 vídeos por sujeito
    random.seed(seed)
    qde_videos = np.shape(conjunto)[0] # 1a dimensão do array conjunto
    ordem = random.sample(range(qde_videos), qde_videos)
    conjunto_shuffled = np.zeros(np.shape(conjunto))
    for vid in range(qde_videos):
    # o vid-ésimo vídeo de conjunto_shuffled será o vid-ésimo vídeo de conjunto
    # de acordo com ordem. Isto é, se o primeiro valor de ordem for 23, o 
    # vid-ésimo vídeo de conjunto_shuffled será o 24o vídeo de conjunto.
        conjunto_shuffled[vid][:][:] = conjunto[ordem[vid]][:][:]
    return conjunto_shuffled, ordem

def remove3s(vetor, freqsampl):
    tempo = 3#segundos
    if len(vetor.shape) == 3:
        vetor = vetor[:, :, freqsampl*tempo:]
    elif len(vetor.shape) == 2:
        vetor = vetor[:, freqsampl*tempo:]
    else:
        print("O vetor não tem nem 2, nem 3 dimensões.")
    return vetor

def preprocessData(data, seed):
# Normaliza amplitude, remapeia canais de 40 para 23, remove 3s iniciais, 
# randomiza vídeos
    data = remove3s(data, 128) # data_preprocessed_python.dat sampled at 128Hz

    data = addNoise.remapDEAP(data)
    data_norm = np.zeros((np.shape(data)))

    print("Normalizing data and randomizing video order...")
    for vid in range(np.shape(data)[0]):
        for chan in range(np.shape(data)[1]):
            data_norm[vid][chan] = normalizaAmp(data[vid][chan])
            
    data_norm_shuf, ordem_vids_shuf = shuffleVids(data_norm, seed)
#    print("Ordem aleatorizada: %s" %(ordem_vids_shuf,))
    
    data_preprocessed = data_norm_shuf

    del data_norm, data_norm_shuf
    return data_preprocessed, ordem_vids_shuf

def openBlink(subject):
######DEPRECATED########
# Essa função serviria para fatorar a função addBlink, separando a parte que
# abre o arquivo de corte desejado. Contudo, ela não foi terminada e não está
# pronta para uso em nenhuma circunstância.

    subject = 'C.EGC' #fixed for testing
    dirs_file = 'dirs.txt' # deve estar mesma pasta de preprocess.py
    
    filename = ''.join(['/cortes/', subject, '_ruidos/corte_blink.txt'])
    
    [dir_data, dir_save, 
     dir_data_ruidos, dir_save_ruidos] = addNoise.open_directories(dirs_file)
    # dir_save_ruidos é onde os arquivos de ruído estão guardados
    #os.chdir(dir_data) #só se os arquivos DEAP.dat n estiverem na mesma pasta
    filename_corte = ''.join([dir_save_ruidos, filename])
    [corte, dims_corte] = addNoise.load_TXT(filename_corte)
    
#    print("O arquivo de corte de ruído a 200Hz tem dimensões: %s" % (dims_corte,))
    corte = addNoise.downsampleCorte(corte, 200, 128)

    for vid in range(dims_corte[0]):
        for chan in range(dims_corte[1]):
            corte_norm[vid][chan] = normalizaAmp(data[vid][chan])
    return corte_norm


def addBlink(data_preprocessed, quantidade, amplitude, subject, noiseType):
    dirs_file = 'dirs.txt' # deve estar mesma pasta de preprocess.py
    filename = ''.join(['/cortes/', subject, '_ruidos/corte_', noiseType, '.txt'])
       
    [dir_data, dir_save, 
     dir_data_ruidos, dir_save_ruidos] = addNoise.open_directories(dirs_file)
    # dir_save_ruidos é onde os arquivos de ruído estão guardados
    #os.chdir(dir_data) #só se os arquivos DEAP.dat n estiverem na mesma pasta
    filename_corte = ''.join([dir_save_ruidos, filename])
    [corte, dims_corte] = addNoise.load_TXT(filename_corte)

#    print("O arquivo de corte de ruído a 200Hz tem dimensões: %s" % (dims_corte,))
    corte = addNoise.downsampleCorte(corte, 200, 128)

    corte = remove3s(corte, 128)

    #Normaliza o corte:
    corte_norm = np.zeros(np.shape(corte))

    for chan in range(np.shape(corte)[0]):
        corte_norm[chan] = normalizaAmp(corte[chan])
#    print("O corte de ruído a 128Hz agora tem dimensões: %s" % (np.shape(corte),))
#    print("O arquivo saudável tem dimensões: %s" % (np.shape(data_preprocessed),))
#    print("Remapeando DEAP 40-channels para 23 canais (Aquis. LPC)...")

#    print("O arquivo saudável formatado tem dimensões: %s" % (np.shape(data_preprocessed),))

    data_noise = addNoise.somaRuido(data_preprocessed, corte_norm, quantidade, amplitude)

    data_noise_norm = np.zeros(np.shape(data_noise))

    for vid in range(np.shape(data_noise)[0]): #por canal
        for chan in range(np.shape(data_noise)[1]): #por canal
            data_noise_norm[vid][chan] = normalizaAmp(data_noise[vid][chan])
    
    return data_noise_norm

if __name__ == '__main__':
    filename = 's01.dat'
    print("Carregando arquivo...")

    data, dims = load_file(filename)

    seed = 1
    data_preprocessed, ordem_vids_shuf = preprocessData(data, seed)

    quantidade = 0.85
    amplitude = 0.95
    subject_ruido = 'C.EGC' #pessoa de quem o ruído será utilizado

    data_noise = addBlink(data_preprocessed, quantidade, amplitude, 
                          subject_ruido)

    video_no = 38

    """
    plt.figure(1)
    plt.plot(data[video_no][0])
    plt.ylim(-25, 25)
    """

    plt.figure(2)
    plt.plot(data_preprocessed[ordem_vids_shuf.index(video_no)][0]) 
    plt.ylim(-1, 1)
    
    plt.figure(3)
    plt.plot(data_noise[ordem_vids_shuf.index(video_no)][0]) 
    plt.ylim(-1, 1)
    
    plt.figure(4)
    #apenas o ruído adicionado em cima de data_preprocessed
    plt.plot(data_preprocessed[ordem_vids_shuf.index(video_no)][0] - 
             data_noise[ordem_vids_shuf.index(video_no)][0])
    plt.ylim(-1, 1)    
    
    plt.show()
    
    
    

