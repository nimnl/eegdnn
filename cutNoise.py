# -*- coding: utf-8 -*- 

from datetime import datetime
import numpy as np
import sys, os, glob
import pdb

def total_seconds(si, sf):
    FMT = '%H:%M:%S'
    tdelta = datetime.strptime(sf, FMT) - datetime.strptime(si, FMT)
    total_seconds = tdelta.seconds
    return total_seconds


def get_titulo(tempo_corte, id, pl):
    if tempo_corte == 74: # tamanho em segundos da captação de blink. 11 amostras
        nome = 'corte_blink'
    elif tempo_corte == 51: # 6 amostras
        nome = 'corte_moveolhos'
    elif tempo_corte == 32: # 3 amostras
        nome = 'corte_engolir'
    elif tempo_corte == 57: # 6 amostras
        nome = 'corte_morder'     
    elif tempo_corte == 53: # 4 amostras
        nome = 'corte_movebracos'
    elif tempo_corte == 15: # nada
        nome = 'corte_inicio'
    elif tempo_corte == 5: # nada
        nome = 'corte_fim'
    else:
        nome = 'corte_nao_identificado'
    # dá pra usar um dicionário para fazer esse case
    return nome
    
def open_directories(txt):
    dirs = open(txt, 'r').read().splitlines() # cria lista com cada linha
    
    for k in range(len(dirs)):
        if dirs[k].startswith('dir_data_ruidos'):
            dir_data_ruidos = dirs[k][19:-1] # pega só o caminho, sem as aspas simples
        elif dirs[k].startswith('dir_save_ruidos'):
            dir_save_ruidos = dirs[k][19:-1]
        #else:
        #    print("Erro na formatação do arquivo de diretórios.\n")

    return dir_data_ruidos, dir_save_ruidos

def corte(arquivo_data_txt, arquivo_tempos, dir_save):
    FREQSAMPL = 200
    
    # arquivo txt data
    arquivo = arquivo_data_txt
    # arquivo txt tempos
    tempos = open(arquivo_tempos).read().splitlines()
    
    print("Lendo arquivo data (Pode demorar um pouco)")
    arquivo = open(arquivo)
    arquivo.readline() #remove cabeçalho
    arquivo = np.loadtxt(arquivo, delimiter=",")
    print("Arquivo data carregado com sucesso\n")


    #formata o nome da pasta como "iniciais.do.voluntario_ruidos"
    pasta = arquivo_data_txt.split('/')[-2]#.split('-')[0][:-3]
    # [-2] pega a pasta do voluntário, pois [-1] (último elemento) é o arquivo .txt
    # a ser cortado.
    # Caso a pasta tivesse formatada com a data, split('-') faria a pasta como
    # uma lista [iniciais.dia, mes, ano, 'hora]. [0] pega iniciais.dia. 
    # Tirando [:-3], fica só iniciais.
    pasta = ''.join([pasta, '_ruidos'])

    if not os.path.exists("%s/cortes/%s" % (dir_save, pasta)):
        # cria pasta para salvar o arquivo cortado:
        os.makedirs("%s/cortes/%s" % (dir_save, pasta))  
        
    ini = 0
    for i in range(0, len(tempos), 2): # de 2 em 2 pega sempre inicio e fim juntos
        tempo_corte = total_seconds(tempos[i], tempos[i+1])

        inicio = ini
        fim = inicio + (tempo_corte*FREQSAMPL) #inicio e fim estão formatados em # de pontos
        ini = fim + 1
        
        titulo = "/cortes/%s/%s.txt" % (pasta, get_titulo(tempo_corte, ID_CORTE, PLAYLIST))

        
        corte = arquivo[inicio:fim]
        corte_path = ''.join([dir_save, titulo])

        open(corte_path, 'w+').close()

        np.savetxt(corte_path, corte, delimiter=",")
        
        print("Corte: %s\nTempo: %d\nIntervalo: %d - %d" % (titulo, tempo_corte, inicio, fim))
        print("%s - %s\n" % (tempos[i], tempos[i+1]))
    print("===========================================================\n")


if __name__ == '__main__':

    tempos = glob.glob("tempos_ruidos.txt")[0] #pega arquivo com os tempos de corte
    
#    edf_data_file = '/home/niago/Documents/PIBITI_2017-2018/Codigos/edfs_data/C.EGC.18-12-2017-13.55.37/ruidos.txt'
    
    dirs_file = 'dirs.txt'
    [dir_data_ruidos, dir_save_ruidos] = open_directories(dirs_file)
    
    all_dirs = [i for i in os.walk(dir_data_ruidos)] #lista de pastas/subdiretorios
    # all_dirs[pasta][0] = 'caminho_da_pasta'
    # all_dirs[pasta][1] = [] #lista vazia
    # all_dirs[pasta][2] = ['arquivos', 'da', 'pasta']
    
    all_dirs = all_dirs[1:] #all_dirs[0] é o diretório root de todas as pastas de edfs

    
    for pasta in range(len(all_dirs)):
        for arquivo in range(len(all_dirs[pasta][2])):
            if all_dirs[pasta][2][arquivo].startswith('ruidos.txt'): #vê se é um arquivo ruidos.txt
                edf_data_file = '/'.join([all_dirs[pasta][0], all_dirs[pasta][2][arquivo]])
#                pdb.set_trace()
                corte(edf_data_file, tempos, dir_save_ruidos)
                print("Corte dos ruídos realizado para %s." % edf_data_file.split('/')[-2])
                #expĺicação para essa formatação, no script de corte
                print("===========================================================\n\n")
            elif 'ruidos.txt' not in all_dirs[pasta][2]: #se não houver ruidos.txt na pasta
                print("Ruidos inexistentes para %s" % all_dirs[pasta][0])
        
    




