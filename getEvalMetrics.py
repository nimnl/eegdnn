import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy import stats
from scipy.signal import butter, lfilter, freqz

def getXCorrelation(comRuido, semRuido, pred):
#Teremos um numero de XCorr para cada canal do conjunto, comparando
#o conjunto predito com o conjunto com ruído e sem ruído através
#dos coeficientes da matrix de correlação
    xcorr_comRuido = np.zeros((np.shape(comRuido)[0], np.shape(comRuido)[2])) #(vids, chans) from [vids][1][chans][points]
    xcorr_semRuido = np.zeros((np.shape(semRuido)[0], np.shape(semRuido)[2]))  #dims(semRuido) must be equal to dims(comRuido)

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

#input2D_comRuido e input2D_semRuido are already the correlation matrices for vid x channel

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
 
    for vid in range(0, np.shape(dados_comRuido)[0]):
        for chan in  range(0, np.shape(dados_comRuido)[2]):
            dataFiltered[vid][0][chan] = butter_bandpass_filter(inputData[vid][0][chan], lowcut, highcut, fs, order=buttorder)

    return dataFiltered

if __name__ == "__main__":

    print("---This is a test execution using random normally distributed variables.---")
    dados_comRuido = np.zeros((40*5, 1, 19, 7680))
    dados_semRuido = np.zeros((40*5, 1, 19, 7680))
    dados_pred = np.zeros((40*5, 1, 19, 7680))
    
    print("Preparing data...")
    for vid in range(0, np.shape(dados_comRuido)[0]):
        for chan in  range(0, np.shape(dados_comRuido)[2]):
            dados_comRuido[vid][0][chan] = np.random.normal(0.5, 0.032, 7680)
            dados_semRuido[vid][0][chan] = np.random.normal(0.4, 0.012, 7680)
            dados_pred[vid][0][chan] = np.random.normal(0.64, 0.055, 7680)
            
    [xcorr_comRuido, xcorr_semRuido] = getXCorrelation(dados_comRuido, dados_semRuido, dados_pred)

    (means_semRuido, stdevs_semRuido,
     means_comRuido, stdevs_comRuido) = getStats(xcorr_semRuido, xcorr_comRuido)
    
    PSNR_comRuido = getPSNR(dados_semRuido, dados_comRuido)
    PSNR_pred = getPSNR(dados_semRuido, dados_pred)
    
    ########################

    dados_filtered = filterData(dados_comRuido)

    plt.plot(dados_filtered[0][0][0], label='Filtered (Butterworth)')

    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.show()
    
    PSNR_butter = getPSNR(dados_semRuido, dados_filtered)
    pdb.set_trace() 
    '''
    plt.figure(1)
    plt.plot(normal)
    plt.figure(2)
    plt.plot(normal2)
    plt.figure(3)
    plt.plot(normal3)

    plt.show()
    '''





