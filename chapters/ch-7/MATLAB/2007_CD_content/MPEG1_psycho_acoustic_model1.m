function [SMR, min_threshold_subband, frame_psd_dBSPL, ...
    masking_threshold] = MPEG1_psycho_acoustic_model1(frame)

% function [SMR, min_threshold_subband, frame_psd_SPL, ...
%    masking_threshold] = MPEG1_psycho_acoustic_model1(frame)
% Computes the masking threshold (in dB) corresponding to psycho-acoustic
% model #1 used in MPEG-1 Audio (cf  ISO/CEI  norm 11172-3:1993 (F), pp.
% 122-128). 
% Input |frame| length should be 512 samples, in the [-1,+1] range. 
% |SMR| returns 27 signal-to-mask ratios (in dB).
% |min_threshold_subband| returns the minimun of |masking threshold| in each
% of the 32 sub-bands. 
% |frame_psd_SPL| returns the estimated PSD of the input frame, in dB SPL,
% assuming the level of full scale signals is set to 96 dB SPL.
%
% Copyright N. Moreau, ENST Paris, 19/03/02
% Modified by Thierry Dutoit, FPMs Mons, 03/05/07

global LTq_i LTq_k Table_z Frontieres_i Frontieres_k Larg_f

if size(LTq_i)==[0,0]
   MPEG1_psycho_acoustic_model1_init;
end;

N = length(frame);
if N ~= 512
    disp('Frame length must be set to 512')
    return
end

% FFT
% ***

hann = sqrt(8/3)/2*[ones(N, 1) - cos(2*pi*(0:N-1)'/N)];
if sum(abs(frame)) > 0
    X1 = fft(frame.*hann);
    X1 = (abs(X1(1:N/2+1)).^2)/N;
    perio_xn_db = 10*log10(X1);
else
    perio_xn_db = zeros(N/2+1,1);
end
%offset = max(perio_xn_db) - 96;
%X = perio_xn_db - offset;

% NB: since the absolute acoustic level set by the listener is not known by
% the MPEG psycho-acoustic model, it assumes that the level is set such
% that  a full-scale signal corresponds to 96 dB SPL. Since 16 bits signals
% have about 96 dB of dynamics, this implies that the LSB is close to 
% the absolute auditory threshold.
% Since the absolute value of input samples is assumed to be <1, a
% full-scale signal, i.e. ones(1:512), will produce a PSD peak at
% 10*log10(512)=27.09dB. Hence the 96-27.09 dB offset.
offset=96-27.09;
X = perio_xn_db + offset;
frame_psd_dBSPL=X(1:256);

% Tonal and noise masker detection 
% *************************

% Local maximum search 

max_local = zeros(250, 1);
for k = 3:250
    if X(k) > X(k-1) & X(k) >= X(k+1)
        max_local(k) = 1;
    end
end

tonal = zeros(250, 1);
for k = 3:62
    if max_local(k)
        tonal(k) = 1;
        for j = [-2 2]
            if X(k) - X(k+j) < 7
                tonal(k) = 0;
            end
        end
    end
end

for k = 63:126
    if max_local(k)
        tonal(k) = 1;
        for j = [-3 -2 2 3]
            if X(k) - X(k+j) < 7
                tonal(k) = 0;
            end
        end
    end
end

for k = 127:250
    if max_local(k)
        tonal(k) = 1;
        for j = [-6:-2 2:6]
            if X(k) - X(k+j) < 7
                tonal(k) = 0;
            end
        end
    end	
end

% Tonal masker detection

X_tm = zeros(250,1);

for k = 1:250
    if tonal(k)
        temp = 10^(X(k-1)/10) + 10^(X(k)/10) + 10^(X(k+1)/10);
        X_tm(k) = 10*log10(temp);
        X(k-1) = -100; 
        X(k)   = -100;
        X(k+1) = -100;
    else
        X_tm(k) = -100;
    end
end

X_nm = -100*ones(250, 1);
k = 1;
for k1 = Frontieres_k
    geom_mean = 1;
    pow = 0;
    raies_en_sb = 0;
    while k <= k1
        geom_mean = geom_mean*k;
        pow = pow + 10^(X(k)/10);
        k = k + 1;
        raies_en_sb = raies_en_sb + 1;
    end
    geom_mean = floor(geom_mean^(1/raies_en_sb));
    X_nm(geom_mean) = 10*log10(pow);
end

X_tm_avant = X_tm;
X_nm_avant = X_nm;

% Decimation of maskers
% *****************

for k = 1:250
    if X_tm(k) < LTq_k(k)
        X_tm(k) = -100;
    end
    if X_nm(k) < LTq_k(k)
        X_nm(k) = -100;
    end
end

% Sliding window for eliminating neigbouring tonal maksers

upper_bound = 1;
lower_bound = 1;
while upper_bound < 250
    [ans, max_ix] = max(X_tm(lower_bound:upper_bound));
    for k = lower_bound:upper_bound
        if k-lower_bound+1 ~= max_ix
            X_tm(k) = -100;
        end
    end
    lower_bound = lower_bound + 1;
    upper_bound = lower_bound + Larg_f(lower_bound);
end

% Individual masking thresholds
% **********************

% "k" to "i"
Nbre_comp_i = length(Table_z);
X_tm_i = -100*ones(Nbre_comp_i, 1);
X_nm_i = -100*ones(Nbre_comp_i, 1);

for k = 1:250
    if X_tm(k) >= -10
        X_tm_i(ppv(k)) = X_tm(k);
    end
end

for k = 1:250
    if X_nm(k) >= -10
        X_nm_i(ppv(k)) = X_nm(k);
    end
end

% Overall masking thresholds
% ********************

seuil_m = zeros(Nbre_comp_i, 1);

no_tm = 0;
no_nm = 0;
for i = 1:Nbre_comp_i
    if X_tm_i(i) > -100
        no_tm = no_tm + 1;
    end
    if X_nm_i(i) > -100
        no_nm = no_nm + 1;
    end
end

tab_tm = zeros(1,no_tm);
tab_nm = zeros(1,no_nm);

ix = 1;
for i = 1:Nbre_comp_i
    if X_tm_i(i) > -100
        tab_tm(ix) = i;
        ix = ix + 1;
    end
end

ix = 1;
for i = 1:Nbre_comp_i
    if X_nm_i(i) > -100
        tab_nm(ix) = i;
        ix = ix + 1;
    end
end

for i = 1:Nbre_comp_i
    sum_tm = 0;
    z_i = Table_z(i);
    for j = tab_tm
        z_j = Table_z(j);
        dz = z_i - z_j;
        if dz >= -3 & dz < 8
            LT_tm = X_tm_i(j) + (-1.525 - 0.275*z_j - 4.5) + vf(dz, j, X_tm_i);
            sum_tm = sum_tm + 10 ^ (LT_tm/10);
        end
    end
    sum_nm = 0;
    for j = tab_nm
        z_j = Table_z(j);
        dz = z_i - z_j;
        if dz >= -3 & dz < 8
            LT_nm = X_nm_i(j) + (-1.525 - 0.175*z_j - 0.5) + vf(dz, j, X_nm_i);
            sum_nm = sum_nm + 10 ^ (LT_nm/10);
        end
    end
    seuil_m(i) = 10 * log10(10^(LTq_i(i)/10) + sum_tm + sum_nm);
end

% Final masking threshold, min masking threshold in each sub-band, and
% signal-to masks 
% ****************************************************

masking_threshold = zeros(1,256);
min_threshold_subband = zeros(1,256);
for i = 1:6
    t1 = seuil_m(8*(i-1)+1:8*(i));
    masking_threshold(8*(i-1)+1:8*(i)) = t1;
    min_threshold_subband(8*(i-1)+1:8*(i)) = ones(1,8)*min(t1);
end
for i = 7:12
    i1 = i - 6;
    t1 = seuil_m(49+4*(i1-1):48+4*(i1));
    t2(1:2:7) = t1;
    t2(2:2:8) = t1;
    masking_threshold(8*(i-1)+1:8*(i)) = t2;
    min_threshold_subband(8*(i-1)+1:8*(i)) = ones(1,8)*min(t1);
end
for i = 13:30
    i1 = i - 12;
    t1 = seuil_m(73+2*(i1-1):72+2*(i1));
    t2(1:4:5) = t1;
    t2(2:4:6) = t1;
    t2(3:4:7) = t1;
    t2(4:4:8) = t1;
    masking_threshold(8*(i-1)+1:8*(i)) = t2;
    min_threshold_subband(8*(i-1)+1:8*(i)) = ones(1,8)*min(t1);
end
for i = 31:32
    masking_threshold(8*(i-1)+1:8*(i)) = ones(1,8)*min(t1);
    min_threshold_subband(8*(i-1)+1:8*(i)) = ones(1,8)*min(t1);
end

% masking_threshold = masking_threshold + offset;
% min_threshold_subband = min_threshold_subband + offset;
for i = 1:32
    SMR(i) = max(frame_psd_dBSPL((i-1)*8+1:i*8)) ...
        - min_threshold_subband(i*8);
end

%==========
% Subfunctions
%==========

function i0 = ppv(k0)

if k0 <= 48
   i0 = k0;
elseif k0 <= 96
   i0 = floor((k0-48)/2) + 48;
else
   i0 = round((k0-96)/4) + 72;
end;
if i0 > 108
   i0 = 108;
end;


function le_vf = vf(dz, j, X)

if dz < -1
   le_vf = 17 * (dz + 1) - (0.4 * X(j) + 6);
elseif dz < 0
   le_vf = (0.4 * X(j) + 6) * dz;
elseif dz < 1
   le_vf = -17 * dz;
else
   le_vf = -(dz - 1) * (17 - 0.15 * X(j)) - 17;
end;
