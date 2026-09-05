
import numpy as np

LTq_i = np.array([])
LTq_k = np.array([])
Table_z = np.array([])
Frontieres_i = np.array([])
Frontieres_k = np.array([])
Larg_f = np.array([])

def MPEG1_psycho_acoustic_model1(frame):
    # function [SMR, min_threshold_subband, frame_psd_SPL, ...
    #    masking_threshold] = MPEG1_psycho_acoustic_model1(frame)
    # Computes the masking threshold (in dB) corresponding to psycho-acoustic
    # model #1 used in MPEG-1 Audio (cf  ISO/CEI  norm 11172-3:1993 (F), pp.
    # 122-128). 
    # Input |frame| length should be 512 samples, in the [-1,+1] range. 
    # |SMR| returns 27 signal-to-mask ratios (in dB).
    # |min_threshold_subband| returns the minimun of |masking threshold| in each
    # of the 32 sub-bands. 
    # |frame_psd_SPL| returns the estimated PSD of the input frame, in dB SPL,
    # assuming the level of full scale signals is set to 96 dB SPL.
    #
    # Copyright N. Moreau, ENST Paris, 19/03/02
    # Modified by Thierry Dutoit, FPMs Mons, 03/05/07

    global LTq_i, LTq_k, Table_z, Frontieres_i, Frontieres_k, Larg_f

    def ppv(k0_new):
        k0 = k0_new+1 #Matlab logic below
        if k0 <= 48:
            i0 = k0
        elif k0 <= 96:
            i0 = np.floor((k0-48)/2) + 48
        else:
            i0 = np.round((k0-96)/4) + 72
        if i0 > 108:
            i0 = 108
        return i0-1 #back to python indexing

    def vf(dz, j, X):
        if dz < -1:
            le_vf = 17 * (dz + 1) - (0.4 * X[j] + 6)
        elif dz < 0:
            le_vf = (0.4 * X[j] + 6) * dz
        elif dz < 1:
            le_vf = -17 * dz
        else:
            le_vf = -(dz - 1) * (17 - 0.15 * X[j]) - 17
        return le_vf

    if len(LTq_i)==0:
        MPEG1_psycho_acoustic_model1_init()

    N = len(frame)
    if N != 512:
        print('Frame length must be set to 512')
        return

    # FFT
    # ***

    hann = np.sqrt(8/3)/2*(np.ones(N) - np.cos(2*np.pi*np.arange(N)/N))
    if np.sum(np.abs(frame)) > 0:
        X1 = np.fft.fft(np.multiply(frame,hann))
        X1 = (np.abs(X1[0 : (N//2) + 1])**2) / N
        perio_xn_db = 10*np.log10(X1)
    else:
        perio_xn_db = np.zeros(int(N/2)+1)
    #offset = max(perio_xn_db) - 96;
    #X = perio_xn_db - offset;

    # NB: since the absolute acoustic level set by the listener is not known by
    # the MPEG psycho-acoustic model, it assumes that the level is set such
    # that  a full-scale signal corresponds to 96 dB SPL. Since 16 bits signals
    # have about 96 dB of dynamics, this implies that the LSB is close to 
    # the absolute auditory threshold.
    # Since the absolute value of input samples is assumed to be <1, a
    # full-scale signal, i.e. ones(1:512), will produce a PSD peak at
    # 10*log10(512)=27.09dB. Hence the 96-27.09 dB offset.

    offset=96-27.09
    X = perio_xn_db + offset
    frame_psd_dBSPL=X[0:256].copy()

    # Tonal and noise masker detection 
    # *************************

    # Local maximum search 

    max_local = np.zeros(250)
    for k in range(2,250):
        if X[k] > X[k-1] and X[k] >= X[k+1]:
            max_local[k] = 1

    tonal = np.zeros(250)
    for k in range(2,62):
        if max_local[k]:
            tonal[k] = 1
            for j in [-2, 2]:
                if X[k] - X[k+j] < 7:
                    tonal[k] = 0

    for k in range(62,126):
        if max_local[k]:
            tonal[k] = 1
            for j in [-3, -2, 2, 3]:
                if X[k] - X[k+j] < 7:
                    tonal[k] = 0

    for k in range(126,250):
        if max_local[k]:
            tonal[k] = 1
            for j in list(range(-6, -1)) + list(range(2, 7)):
                if X[k] - X[k+j] < 7:
                    tonal[k] = 0

    # Tonal masker detection

    X_tm = np.zeros(250)
    for k in range(250):
        if tonal[k]:
            temp = 10**(X[k-1]/10) + 10**(X[k]/10) + 10**(X[k+1]/10)
            X_tm[k] = 10*np.log10(temp)
            X[k-1] = -100
            X[k]   = -100
            X[k+1] = -100
        else:
            X_tm[k] = -100

    X_nm = -100*np.ones(250)
    k = 0
    for k1 in Frontieres_k:
        geom_mean = 1
        pow = 0
        raies_en_sb = 0
        while k <= k1-1:
            geom_mean = geom_mean*(k+1)
            pow = pow + 10**(X[k]/10)
            k = k + 1
            raies_en_sb = raies_en_sb + 1

        geom_mean = int(np.floor(geom_mean**(1/raies_en_sb)))
        X_nm[geom_mean-1] = 10*np.log10(pow)

    X_tm_avant = X_tm
    X_nm_avant = X_nm

    # Decimation of maskers
    # *****************

    for k in range(250):
        if X_tm[k] < LTq_k[k]:
            X_tm[k] = -100
        if X_nm[k] < LTq_k[k]:
            X_nm[k] = -100

    # Sliding window for eliminating neigbouring tonal maksers

    upper_bound = 0
    lower_bound = 0
    while upper_bound < 249:
        max_ix = np.argmax(X_tm[int(lower_bound):int(upper_bound)+1])
        for k in range(int(lower_bound), int(upper_bound) + 1):
            if k-lower_bound != max_ix:
                X_tm[k] = -100
        lower_bound = lower_bound + 1
        upper_bound = lower_bound + Larg_f[lower_bound]

    # Individual masking thresholds
    # **********************

    # "k" to "i"
    Nbre_comp_i = len(Table_z)
    X_tm_i = -100*np.ones(Nbre_comp_i)
    X_nm_i = -100*np.ones(Nbre_comp_i)

    for k in range(250):
        if X_tm[k] >= -10:
            X_tm_i[int(ppv(k))] = X_tm[k]

    for k in range(250):
        if X_nm[k] >= -10:
            X_nm_i[int(ppv(k))] = X_nm[k]

    # Overall masking thresholds
    # ********************

    seuil_m = np.zeros(Nbre_comp_i)

    no_tm = 0
    no_nm = 0
    for i in range(Nbre_comp_i):
        if X_tm_i[i] > -100:
            no_tm = no_tm + 1

        if X_nm_i[i] > -100:
            no_nm = no_nm + 1

    tab_tm = np.zeros(no_tm)
    tab_nm = np.zeros(no_nm)

    ix = 0
    for i in range(Nbre_comp_i):
        if X_tm_i[i] > -100:
            tab_tm[ix] = i
            ix = ix + 1

    ix = 0
    for i in range(Nbre_comp_i):
        if X_nm_i[i] > -100:
            tab_nm[ix] = i
            ix = ix + 1

    for i in range(Nbre_comp_i):
        sum_tm = 0
        z_i = Table_z[i]
        for j in tab_tm:
            z_j = Table_z[int(j)]
            dz = z_i - z_j
            if dz >= -3 and dz < 8:
                LT_tm = X_tm_i[int(j)] + (-1.525 - 0.275*z_j - 4.5) + vf(dz, int(j), X_tm_i)
                sum_tm = sum_tm + 10**(LT_tm/10)

        sum_nm = 0
        for j in tab_nm:
            z_j = Table_z[int(j)]
            dz = z_i - z_j
            if dz >= -3 and dz < 8:
                LT_nm = X_nm_i[int(j)] + (-1.525 - 0.175*z_j - 0.5) + vf(dz, int(j), X_nm_i)
                sum_nm = sum_nm + 10**(LT_nm/10)

        seuil_m[i] = 10 * np.log10(10**(LTq_i[i]/10) + sum_tm + sum_nm)

    # Final masking threshold, min masking threshold in each sub-band, and
    # signal-to masks 
    # ****************************************************

    masking_threshold = np.zeros(256)
    min_threshold_subband = np.zeros(256)
    for i in range(6):
        t1 = seuil_m[8*i:8*(i+1)]
        masking_threshold[8*i:8*(i+1)] = t1
        min_threshold_subband[8*i:8*(i+1)] = np.min(t1)

    t2 = np.zeros(8)
    for i in range(6,12):
        i1 = i - 6
        t1 = seuil_m[48+4*i1:48+4*(i1+1)]
        t2[0:7:2] = t1
        t2[1:8:2] = t1
        masking_threshold[8*i:8*(i+1)] = t2
        min_threshold_subband[8*i:8*(i+1)] = np.min(t1)

    for i in range(12,30):
        i1 = i - 12
        t1 = seuil_m[72+2*i1:72+2*(i1+1)]
        t2 = np.zeros(8)
        t2[0:5:4] = t1
        t2[1:6:4] = t1
        t2[2:7:4] = t1
        t2[3:8:4] = t1
        masking_threshold[8*i:8*(i+1)] = t2
        min_threshold_subband[8*i:8*(i+1)] = np.min(t1)

    for i in range(30,32):
        masking_threshold[8*i:8*(i+1)] = np.min(t1)
        min_threshold_subband[8*i:8*(i+1)] = np.min(t1)

    # masking_threshold = masking_threshold + offset
    # min_threshold_subband = min_threshold_subband + offset
    SMR = np.zeros(32)
    for i in range(32):
        SMR[i] = np.max(frame_psd_dBSPL[i*8:(i+1)*8])- min_threshold_subband[(i+1)*8-1]
    return SMR,min_threshold_subband,frame_psd_dBSPL,masking_threshold

'''
Constants used in the MPEG-1 psycho-acoustic model #1
(for Fs=44100 Hz)

Copyright N. Moreau, ENST Paris, 19/3/02
'''

def MPEG1_psycho_acoustic_model1_init():
    global LTq_i, LTq_k, Table_z, Frontieres_i, Frontieres_k, Larg_f

    # "i" to "k"
    i1 = np.arange(1,49)
    i2 = np.arange(49,73)-48 
    i3 = np.arange(73,109)-72
    i_to_k = np.concatenate((i1,2*i2+48,4*i3+96)) #49+25+36=120 elements
    # Absolute auditory threshold as a function of "i"
    LTq_i = np.array([25.87, 14.85, 10.72,  8.50,  7.10,  6.11,  5.37,  4.79,
                4.32,  3.92,  3.57,  3.25,  2.95,  2.67,  2.39,  2.11,
                1.83,  1.53,  1.23,  0.90,  0.56,  0.21, -0.17, -0.56,
                -0.96, -1.38, -1.79, -2.21, -2.63, -3.03, -3.41, -3.77,
                -4.09, -4.37, -4.60, -4.78, -4.91, -4.97, -4.98, -4.92,
                -4.81, -4.65, -4.43, -4.17, -3.87, -3.54, -3.19, -2.82,
                -2.06, -1.32, -0.64, -0.04,  0.47,  0.89,  1.23,  1.51,
                1.74,  1.93,  2.11,  2.28,  2.46,  2.63,  2.82,  3.03,
                3.25,  3.49,  3.74,  4.02,  4.32,  4.64,  4.98,  5.35,
                6.15,  7.07,  8.10,  9.25, 10.54, 11.97, 13.56, 15.31,
                17.23, 19.34, 21.64, 24.15, 26.88, 29.84, 33.05, 36.52,
                40.25, 44.27, 48.59, 53.22, 58.18, 63.49, 68.00, 68.00,
                68.00, 68.00, 68.00, 68.00, 68.00, 68.00, 68.00, 68.00,
                68.00, 68.00, 68.00, 68.00])

    # Absolute auditory threshold as a function of "k"
    LTq_k = np.zeros(250)
    for i in range(len(LTq_i)):
        LTq_k[i_to_k[i]-1] = LTq_i[i]

    last_nonzero=0
    for k in range(250):
        if LTq_k[k] == 0:
            LTq_k[k] = last_nonzero
        else:
            last_nonzero = LTq_k[k]

    # axe_freq = (0:249)*Fe/512/1000
    # figure(1); hold off
    # plot(axe_freq, LTq_k); hold on
    # axis([0 20 -10 70])

    # Bark frequencies as a fucntion of "i"
    Table_z = np.array([ .850, 1.694, 2.525, 3.337, 4.124, 4.882, 5.608, 6.301,
                6.959,  7.581,  8.169,  8.723,  9.244,  9.734, 10.195, 10.629,
                11.037, 11.421, 11.783, 12.125, 12.448, 12.753, 13.042, 13.317,
                13.578, 13.826, 14.062, 14.288, 14.504, 14.711, 14.909, 15.100,
                15.284, 15.460, 15.631, 15.796, 15.955, 16.110, 16.260, 16.406,
                16.547, 16.685, 16.820, 16.951, 17.079, 17.205, 17.327, 17.447,
                17.680, 17.905, 18.121, 18.331, 18.534, 18.731, 18.922, 19.108,
                19.289, 19.464, 19.635, 19.801, 19.963, 20.120, 20.273, 20.421,
                20.565, 20.705, 20.840, 20.972, 21.099, 21.222, 21.342, 21.457,
                21.677, 21.882, 22.074, 22.253, 22.420, 22.576, 22.721, 22.857,
                22.984, 23.102, 23.213, 23.317, 23.415, 23.506, 23.592, 23.673,
                23.749, 23.821, 23.888, 23.952, 24.013, 24.070, 24.125, 24.176,
                24.225, 24.271, 24.316, 24.358, 24.398, 24.436, 24.473, 24.508,
                24.542, 24.574, 25, 25])
        
    Frontieres_i = np.array([1, 2, 3, 5, 6, 8, 9, 11, 13, 15, 17, 20, 23, 27, 32, 37,
            45, 50, 55, 61, 68, 75, 81, 93, 106])

    Frontieres_k = np.zeros(len(Frontieres_i))
    for i in range(len(Frontieres_i)):
        Frontieres_k[i] = i_to_k[Frontieres_i[i]-1]

    # Bandwidth of critical bands
    f_250_c = np.concatenate(([0], Frontieres_k, [296]))
    f_250_d = np.concatenate(([0], Frontieres_k, [256]))

    upper_bound = 1
    lower_bound = 1
    Larg_f = np.zeros(224)
    while upper_bound < 250:
        lower_bound = lower_bound + 1
        larg_bas = 0
        no_ech_bas = 0
        for k in range(25):
            if lower_bound >= f_250_c[k] and lower_bound < f_250_c[k+1]:
                larg_bas = f_250_c[k+1] - f_250_c[k]
                no_ech_bas = f_250_c[k+1] - lower_bound

        if no_ech_bas >= np.ceil(larg_bas/2):
            larg_fen = np.ceil(larg_bas/2)
        else:
            larg_haut = 0
            no_ech_haut = 0
            for k in range(25):
                if upper_bound >= f_250_c[k] and upper_bound < f_250_c[k+1]:
                    larg_haut = f_250_c[k+1] - f_250_c[k]
                    no_ech_haut = upper_bound - f_250_c[k]

            no_ech_tot = no_ech_haut + no_ech_bas
            larg_fen = np.ceil((larg_bas*no_ech_bas/no_ech_tot+larg_haut*no_ech_haut/no_ech_tot)/2)

        upper_bound = lower_bound + larg_fen
        Larg_f[lower_bound-1] = larg_fen
    return 
