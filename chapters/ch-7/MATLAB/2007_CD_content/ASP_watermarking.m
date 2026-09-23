%% Chapter 7 - How could music contain hidden information
% This is a companion file to the book "Applied Signal Processing", 
% by T.Dutoit and F. Marques, Springer 2008.
% 
% It is supposed to be run cell-by-cell, using the cell mode of 
% MATLAB 6 and later versions. Search for "what are cells" in the 
% product help of MATLAB to know how to use them.
% 
% This file uses the SIGNAL_PROCESSING toolbox of MATLAB.
 
%%
% In the following script, we will develop watermarking systems,
% whose design is based on classical communication systems using spread
% spectrum modulation. We first examine the implementation of a
% watermarking system and give a brief overview of its performance in the
% simple and theoretical configuration where the audio signal is a white
% Gaussian noise (Section 1). Then, we show how this system can be adapted
% to account for the audio signal specificities, while still satisfying the
% major properties and requirements of watermarking applications (namely
% inaudibility, correct detection and robustness of the watermark): we
% extend the initial system by focussing on the correct detection of the
% watermark (Section 2), on the inaudibility constraint (Section 3), and on
% the robustness of the system to MP3 compression (Section 4). 
%
% Copyright C. Baras, N. Moreau, T. Dutoit (2007)

clear all;
set(0, 'defaultFigureColor', 'w');

%% 1. Audio watermarking, seen as a digital communication problem
% The following digital communication system is the direct implementation
% of the theoretical results detailed in Section 1 of the book chapter. The
% watermark message is embedded in the audio signal at the binary rate
% |R|=100 bps. To establish the analogy between watermarking system and
% communication channel, the audio signal is viewed as the channel noise.
% In this section, it will therefore be modeled as a white Gaussian noise
% with zero mean and variance |sigma2_x|. |sigma2_x| is computed so that
% the Signal-to-Noise Ratio (SNR), i.e. the ratio between the watermark and
% the audio signal powers, equals -20 dB.

%% 1.1. Emitter/Embedder
% Let us first generate a watermark message, and its corresponding bit
% sequence.

message = 'Audio watermarking: this message will be embedded in an audio signal, using spread spectrum modulation, and later retrieved from the modulated signal.'
bits = dec2bin(message); % converting each ascii character to a 7-bit string
bits = bits(:)'; % reshaping to a single string of 1's and 0's
bits = double(bits) - 48; % converting to a (1x1050) array of 0's and 1's
symbols = bits*2-1;  % a (1x1050) array of -1's and +1's
N_bits = length(bits);

%%
% We then generate the audio signal, the spread spectrum sequence, and the
% watermarked signal. The latter is obtained by first deriving a symbol
% sequence (-1's and +1's) from the input bit sequence (0's and +1's), and
% then by modulating the spread spectrum signal by the symbol sequence. This
% is achieved by concatenating spread spectrum waveforms weighted by the
% symbol values. 

% Random sequences generators initialization
rand('seed', 12345);
randn('seed', 12345);

% Sampling frequency |Fs|, binary rate |R|, samples per bit |N_samples| and |SNR|
Fs = 44100;
R = 100;
N_samples = round(Fs/R);
SNR_dB = -20;

% Spread spectrum waveform: a random sequence of -1's and +1's
spread_waveform = 2*round( rand(N_samples, 1) ) - 1; 

% Audio signal: a Gaussian white noise with fixed variance
sigma2_x = 10^(-SNR_dB/10);
audio_signal = sqrt(sigma2_x)*randn(N_bits*N_samples, 1);

% Modulated signal and watermark signal
modulated_signal = zeros(N_bits*N_samples, 1);
for m = 0:N_bits-1
     modulated_signal(m*N_samples+1:m*N_samples+N_samples) = ...
         symbols(m+1)*spread_waveform;
end
watermark_signal = modulated_signal; % no gain applied

% Plotting the baseband signal (i.e., the emitted signal corresponding
% to the input bit sequence) and the watermark signal

emitted_signal = ones(N_samples, 1)*symbols;
emitted_signal = emitted_signal(:);

clf;
axe_ech = (4*N_samples:7*N_samples+1);
subplot(2,1,1); 
plot(axe_ech/Fs, emitted_signal(axe_ech));
axis([axe_ech(1)/Fs axe_ech(end)/Fs -1.1 1.1]);
xlabel('Time(s)');
subplot(2,1,2); 
plot(axe_ech/Fs, watermark_signal(axe_ech));
axis([axe_ech(1)/Fs axe_ech(end)/Fs -1.1 1.1]);
xlabel('Time(s)');

%% 
% As expected, the power spectral density (PSD) of the spread spectrum
% sequence (i.e., the watermark signal) is much flatter than that of the
% baseband signal (i.e., the emitted signal) corresponding to the input
% bit sequence, while their power is identical: 0dB.

power_baseband_dB=10*log10(var(emitted_signal))
power_spread_spectrum_dB=10*log10(var(watermark_signal))

close(1);
clf; 
subplot(2,1,1)
% NB: we claim Fs=2 so as to make pwelch return |FFT²/N|, in which the
% variance of a white noise is directly readable. See the appendix in
% ASP_audio_cd.m for more details.
pwelch(emitted_signal,[],[],[],2);
xlabel('Normalized Frequency : f/(Fs/2)');
subplot(2,1,2)
pwelch(watermark_signal,[],[],[],2);
xlabel('Normalized Frequency : f/(Fs/2)');

%%
% The embedding process finally consists in adding the result to the audio
% signal, yielding the audio watermarked signal.

watermarked_signal = watermark_signal + audio_signal;

% Plotting the first 100 samples of the watermark, audio, and watermarked
% signals shows that the watermarked signal is only slightly different from
% the audio signal, given the SNR we have imposed.
clf;
subplot(3,1,1); plot((1:100)/Fs, watermark_signal(1:100), 'k-');
set(gca,'ylim',[-1.1 1.1]);
subplot(3,1,2); plot((1:100)/Fs, audio_signal(1:100));
subplot(3,1,3); plot((1:100)/Fs, watermarked_signal(1:100), 'r');
xlabel('Time(s)');

%% 1.2. Receiver
% The watermark receiver is a correlation demodulator. It computes the
% normalized scalar product |alpha| between frames of the received signal
% and the spread spectrum waveform, and decides on the received bits, based
% to the sign of |alpha|. One can see on the plot (for bits 81 to 130) that
% the watermark sometimes imposes a wrong sign to |alpha|, leading to
% erroneous bit detection.

alpha = zeros(N_bits, 1);
received_bits = zeros(1,N_bits);
for m = 0:N_bits-1
    alpha(m+1) = (watermarked_signal(m*N_samples+1:m*N_samples+N_samples)' ...
        * spread_waveform)/N_samples;
    if alpha(m+1) <= 0
        received_bits(m+1) = 0;
    else
        received_bits(m+1) = 1;
    end
end

% Plotting the input bits, alpha, and the received bits.
clf;
range=(81:130);
subplot(3,1,1); 
stairs(range, bits(range));
set(gca,'ylim',[-0.1 1.1]);
subplot(3,1,2); 
stairs(range, alpha(range));
set(gca,'ylim',[-3 3]);
subplot(3,1,3); 
stairs(range, received_bits(range));
set(gca,'ylim',[-0.1 1.1]);
xlabel('index');

%%
% The performance of the receiver can be estimated by computing the the Bit
% Error Rate (BER) and by decoding the received message. As can be observed
% on the message, a bit error rate of .02 is disastrous in terms of message
% understandability.

number_of_erroneous_bits= sum(bits ~= received_bits)
total_number_of_bits=N_bits
BER = number_of_erroneous_bits/N_bits

received_chars = reshape(received_bits(1:N_bits),N_bits/7, 7);
received_message = char( bin2dec(num2str(received_chars)))'

%% 1.3. System performance overview
% A good overview of the system performance can be drawn from the
% statistical observation of |alpha| values, through a histogram plot. As
% expected, a bimodal distribution is found, with non zero overlap.
% Notice that the histogram only gives a rough idea of the underlying 
% distribution, since the number of emitted bits is small. 

clf;
hist(alpha,50);

%%
% In our particular configuration, the spread waveform is a realization of
% a random sequence with values in {+1,-1}, and the audio-noise vectors
% are modeled by |N_samples| independent Gaussian random variables with
% zero mean and variance |sigma2_x|. In this case, it can be shown that the
% Probability Density Function (PDF) of |alpha| is the average of two
% normal distributions with mean |a*g| (|a|=+1 or -1) and variance
% |sigma2_x/N_samples|. 
% Plotting the theoretical normal distribution shows a better view of the
% overlap area observed on the histogram.  In particular, the area of the
% overlap between the two Gaussian modes (0.45) corresponds
% to the theoretical BER (the exact expression of this BER is given in the
% main text). 

alpha_range = -3:.1:3;
gauss0 = 1/2*exp(-((alpha_range-1).^2)/(2*sigma2_x/N_samples)) / sqrt(2*pi*sigma2_x/N_samples);
gauss1 = 1/2*exp(-((alpha_range+1).^2)/(2*sigma2_x/N_samples)) / sqrt(2*pi*sigma2_x/N_samples);
theoretical_PDF=gauss0+gauss1;

plot(alpha_range, gauss0); hold on
plot(alpha_range, gauss1);

ind0 = find(alpha_range<=0);
ind1 = find(alpha_range>=0);
ae1 = area(alpha_range(ind1), gauss1(ind1), 'FaceColor', [0.39 0.47 0.64]);   hold on;
ae2 = area(alpha_range(ind0), gauss0(ind0), 'FaceColor', [0.39 0.47 0.64]);

theoretical_BER1=2*sum(gauss1(ind1))*(alpha_range(2)-alpha_range(1))
 
%% 2. Informed watermarking with error-free detection
% The previous system did not take the audio signal specificities into
% account (since audio was modeled as white Gaussian noise). We now 
% examine how to modify the system design to reach one major requirement
% for a watermarking application: that of detecting the watermark message
% without error. For this purpose, the watermark gain (also called
% embedding strength) has to be adjusted to the local audio variations.
% This is referred to as informed watermarking.

% Preparing the data, as in Section 1.
message = 'Audio watermarking: this message will be embedded in an audio signal, using spread spectrum modulation, and later retrieved from the modulated signal.'
bits = dec2bin(message); % converting each ascii character to a 7-bit string
bits = bits(:)'; % reshaping to a single string of 1's and 0's
bits = double(bits) - 48; % converting to a (1x1050) array of 0's and 1's
symbols = bits*2-1;  % a (1x1050) array of -1's and +1's
N_bits = length(bits);
rand('seed', 12345);
randn('seed', 12345);
Fs = 44100;
R = 100;
N_samples = round(Fs/R);
SNR_dB = -20;
spread_waveform = 2*round( rand(N_samples, 1) ) - 1; % in {-1,1}
modulated_signal = zeros(N_bits*N_samples, 1);
for m = 0:N_bits-1
     modulated_signal(m*N_samples+1:m*N_samples+N_samples) = ...
         symbols(m+1)*spread_waveform;
end

% From now on, the audio signal will be a violin signal sampled at
% |Fs|=44.100 Hz, of which only the first |N_bits*N_samples| samples will
% be watermarked. This signal is normalized in [-1,+1].
audio_signal = wavread('violin.wav', [1 N_bits*N_samples]);

soundsc(audio_signal,44100);

%% 2.1. Informed emitter
% To reach a zero BER, the watermark gain is adapted so that the
% correlation between the watermarked audio signal and the spread sequence 
% is at least equal to some security margin |Delta_g| (if |am|=+1)
% and |-Delta_g| (if |am|=-1). |Delta_g| sets up a robustness margin
% against additive perturbation (such as the additive noise introduced by 
% MPEG compression of the watermarked audio signal).
% In the following MATLAB implementation, |Delta_g| is empirically 
% set to 0.005. 

Delta_g = 0.005;
for m=0:N_bits-1
    beta = audio_signal(m*N_samples+1 : m*N_samples+N_samples)' * ...
        spread_waveform /N_samples;
    if symbols(m+1) == 1
        if  beta >= Delta_g
            gain(m+1) = 0;
        else
            gain(m+1) = Delta_g - beta;
        end
    else % if symbols(m+1) == -1
        if  beta <= -Delta_g
            gain(m+1) = 0;
        else
            gain(m+1) = Delta_g + beta;
        end
    end
    watermark_signal(m*N_samples+1 : m*N_samples+N_samples,1) = ...
        gain(m+1)*modulated_signal(m*N_samples+1:m*N_samples+N_samples);
end

watermarked_signal = watermark_signal + audio_signal; % Watermarked audio signal

%%
% Plotting the gain for one second of signal, together with the resulting
% watermark signal and the audio signal, shows that the encoder has to
% make important adjustments as a function of the audio signal.

clf;
subplot(2,1,1);
plot((0:99)*N_samples/Fs, gain(1:100));
subplot(2,1,2);
plot((0:Fs-1)/Fs, watermark_signal(1:Fs));
xlabel('time (s)');

%%
% The watermark, though, is small compared to the audio signal.

clf;
plot((0:Fs-1)/Fs, audio_signal(1:Fs)); 
hold on;
plot((0:Fs-1)/Fs, watermark_signal(1:Fs), 'r-');
xlabel('time (s)');
legend('audio signal','watermark signal');

%%
% Plotting again a few samples of the watermark, audio, and watermarked
% signals shows that the watermarked signal sometimes differs
% significantly from the audio signal. This is confirmed by a listening
% test: the watermark is audible.

clf;
range=(34739:34938);
subplot(3,1,1); plot(range/Fs, watermark_signal(range), 'k-');
set(gca,'xlim',[range(1)/Fs range(end)/Fs]);
subplot(3,1,2); plot(range/Fs, audio_signal(range));
set(gca,'xlim',[range(1)/Fs range(end)/Fs]);
subplot(3,1,3); plot(range/Fs, watermarked_signal(range), 'r');
set(gca,'xlim',[range(1)/Fs range(end)/Fs]);
xlabel('Time(s)');

soundsc(watermarked_signal,Fs);

%% 2.2. Receiver
% Using the same correlation demodulator as in Section 1, we conclude that the
% transmission is now effectively error-free, as confirmed by the resulting
% BER.

for m = 0:N_bits-1
    alpha(m+1) = (watermarked_signal(m*N_samples+1:m*N_samples+N_samples)' ...
        * spread_waveform)/N_samples;
    if alpha(m+1) <= 0
        received_bits(m+1) = 0;
    else
        received_bits(m+1) = 1;
    end
end

% Plotting the input bits, the value of aplha, and the received bits.
clf;
range=(81:130);
subplot(3,1,1); 
stairs(range, bits(range));
set(gca,'ylim',[-0.1 1.1]);
subplot(3,1,2); 
stairs(range, alpha(range));
set(gca,'ylim',[-.01 .01]);
subplot(3,1,3); 
stairs(range, received_bits(range));
set(gca,'ylim',[-0.1 1.1]);
xlabel('frame index');

number_of_erroneous_bits= sum(bits ~= received_bits)
total_number_of_bits=N_bits
BER = number_of_erroneous_bits/total_number_of_bits

received_chars = reshape(received_bits(1:N_bits),N_bits/7, 7);
received_message = char( bin2dec(num2str(received_chars)))'

%% 3. Informed watermarking made inaudible
% The informed watermarking system exposed in Section 2 proves that prior
% knowledge of the audio signal can be efficiently used in the watermarking
% process:  error-free transmission is reached thanks to an adaptive
% embedding gain. However, the resulting watermark is audible, since no
% perceptual condition is imposed on the embedding gain. In this Section,
% we examine how psychoacoustics can be put to profit to ensure the
% inaudibility constraint.
% A psychoacoustic model is used, which provides a signal-dependent masking
% threshold used as an upper bound for the PSD of the  watermark. The
% watermark gain is therefore replaced by an all-pole perceptual shaping
% filter and the reception process is composed of a zero-forcing equalizer,
% fowllowed by a linear-phase Wiener filter.

% Preparing the data, as in Section 2.
message = 'Audio watermarking: this message will be embedded in an audio signal, using spread spectrum modulation, and later retrieved from the modulated signal.'
bits = dec2bin(message); % converting each ascii character to a 7-bit string
bits = bits(:)'; % reshaping to a single string of 1's and 0's
bits = double(bits) - 48; % converting to a (1x1050) array of 0's and 1's
symbols = bits*2-1;  % a (1x1050) array of -1's and +1's
N_bits = length(bits);
rand('seed', 12345);
randn('seed', 12345);
Fs = 44100;
R = 100;
N_samples = round(Fs/R);
SNR_dB = -20;
spread_waveform = 2*round( rand(N_samples, 1) ) - 1; % in {-1,1}
modulated_signal = zeros(N_bits*N_samples, 1);
for m = 0:N_bits-1
     modulated_signal(m*N_samples+1:m*N_samples+N_samples) = ...
         symbols(m+1)*spread_waveform;
end
audio_signal = wavread('violin.wav', [1 N_bits*N_samples]);

soundsc(audio_signal,Fs);

%% 3.1. Emitter, based on perceptual shaping filtering 
% The perceptual shaping filter is an auto-regressive filter with 50
% coefficients |ai| and gain |b0|. It is designed so that the PSD of the
% watermark (obtained by filtering the |modulated_signal|) equals the
% |masking_threshold|. The coefficients of the filter are obtained as in
% Chapter 1, via the Levinson algorithm. Both the masking threshold and the
% shaping filter have to be updated each time the (statistical) properties
% of the |audio_signal| change (here every 512 samples).   

%%
% Let us first compute the masking threshold and apply the associated shaping
% filter to one audio frame (the 11-th frame, for instance)

N_coef = 50;
N_samples_PAM = 512;
PAM_frame = (10*N_samples_PAM+1 : 10*N_samples_PAM+N_samples_PAM);

% Showing the audio signal frame
clf;
plot(PAM_frame/Fs,audio_signal(PAM_frame) );
xlabel('Time (s)')

%%
% Comparing the periodogram of the audio signal, the masking threshold and
% the frequency response of the filter shows that the filter closely
% matches the masking threshold. 
% Even the absolute amplitude level of the filter is the same as that of the
% maksing threshold. As a matter of fact, since the audio 
% signal is normalized in [-1,+1], its nominal PSD is 0 dB.
%
% *MATLAB functions involved:*
%
% * |masking_threshold = psychoacoustical_model( audio_signal )| returns
% the masking threshold deduced from a psychoacoustical analysis of
% the audio vector. 
%
% This implementation is derived from the psycho-acoustic model #1 used in
% MPEG-1 Audio (see ISO/CEI norm 11172-3:1993 (F), pp. 122-128 or MATLAB
% function MPEG1_psycho_acoustic_model1.m from Chapter 3). It is based on
% the same principles as those used in the MPEG model, but it is further
% adapted here so as to make it robust to additive noise (which is a
% specific constraint of watermarking and is not found in MPEG).
%
% * |[b0, ai] = shaping_filter_design(desired_frequency_response_dB, N_coef)| 
% computes the coefficients of an auto-regressive filter:  
%
%                                       b0
% G(z) = ----------------------------------------------
%                           -1                           -N_coef+1
%         ai(1) + ai(2)z    + ...  + ai(N_coef)z
%
% (with ai(1)=1) from the modulus of its |desired_frequency_response|
% (in dB) and the order |N_coef|. 
% The coefficients are obtained as follows: if zero-mean and unity variance
% noise is provided at the input of the filter, the PSD of its ouput is
% given by |desired_frequency_response|. Setting the coefficients so that 
% this PSD best matches |desired_frequency_response| is thus obtained by
% applying the Levinson algorithm to the autocorrelation coefficients of
% the output signal (computed itself from the IFFT of the
% |desired_frequency_response|). 

masking_threshold = psychoacoustical_model(audio_signal(PAM_frame));
shaping_filter_response = masking_threshold;
[b0, ai] = shaping_filter_design(shaping_filter_response, N_coef);

% Plotting results. For details on how we use |pwelch|, see the appendix of
% ASP_audio_cd.m, the companion script file of Chapter 2.
close(1)
pwelch(audio_signal(PAM_frame),[],[],[],2);
hold on;
[H,W]=freqz(b0,ai,256);
stairs(W/pi,masking_threshold,'k--');
plot(W/pi,20*log10(abs(H)),'r','linewidth',2);
hold off;
xlabel('Normalized Frequency : f/(Fs/2)');
legend('Audio signal', 'Masking threshold', 'Filter response', 1);

%%
% Applying this perceptual shaping procedure to the whole watermark signal
% requires to process the audio signal block per block. Notice that the
% filtering continuity from one block to another is ensured by the |state|
% vector, which stores the final state of the filter at the end of one block
% and applies it as initial conditions for the next block. 

state = zeros(N_coef, 1);
for m = 0:fix(N_bits*N_samples/N_samples_PAM)-1
    PAM_frame = (m*N_samples_PAM+1:m*N_samples_PAM+N_samples_PAM);
    
    % Shaping filter design
    masking_threshold = psychoacoustical_model( audio_signal(PAM_frame) );
    shaping_filter_response = masking_threshold;
    [b0, ai] = shaping_filter_design(shaping_filter_response, N_coef);
    
    % Filtering stage
    [watermark_signal(PAM_frame,1), state] = ...
        filter(b0, ai, modulated_signal(PAM_frame), state);
end

% Filtering the last, incomplete frame in |watermark_signal|
PAM_frame = (m*N_samples_PAM+N_samples_PAM+1:N_bits*N_samples);
watermark_signal(PAM_frame,1) = ...
    filter(b0, ai, modulated_signal(PAM_frame), state);

plot((0:Fs-1)/Fs, audio_signal(1:Fs)); 
hold on;
plot((0:Fs-1)/Fs, watermark_signal(1:Fs), 'r-');
xlabel('time (s)');
legend('audio signal','watermark signal');

%%
% The watermarked audio signal is still obtained by adding the audio signal
% and the watermark signal. Plotting again few samples of the watermark,
% audio, and watermarked signals shows that the watermark signal has now
% been filtered. As a result, the watermark is quite inaudible, as confirmed by a
% listening test. Its level, though, is similar to (if not higher than) that of
% the watermark signal in Section 2.

watermarked_signal = watermark_signal + audio_signal;

clf;
range=(34739:34938);
subplot(3,1,1); plot(range/Fs, watermark_signal(range), 'k');
set(gca,'xlim',[range(1)/Fs range(end)/Fs]);
subplot(3,1,2); plot(range/Fs, audio_signal(range));
set(gca,'xlim',[range(1)/Fs range(end)/Fs]);
subplot(3,1,3); plot(range/Fs, watermarked_signal(range), 'r');
set(gca,'xlim',[range(1)/Fs range(end)/Fs]);
xlabel('Time(s)');

soundsc(watermarked_signal,Fs);

%% 3.2 Receiver, based on zero-forcing equalization and detection
% The zero-forcing equalization aims at reversing the watermark
% perceptual shaping, before extracting the embedded message. The shaping
% filter with frequency response |shaping_filter_response| is therefore
% recomputed from the audio |watermarked_signal|, since the original
% |audio_signal| is not available at the receiver. This process follows the
% same block processing as the watermark synthesis, but involves a moving
% average filtering stage: filtering the |watermarked_signal| by the filter
% whose frequency response is the inverse of the |shaping_filter_response|
% yields the filtered received signal denoted by |equalized_signal|. 
%
% Plotting the PSD of the |equalized_signal| shows its flat spectral
% envelope (hence its name).

equalized_signal = zeros(N_bits*N_samples, 1);        
state = zeros(N_coef, 1);

for m = 0:fix(N_bits*N_samples/N_samples_PAM)-1
    
    PAM_frame = m*N_samples_PAM+1:m*N_samples_PAM+N_samples_PAM;
    
    % Shaping filter design, based the watermarked signal
    masking_threshold = psychoacoustical_model(watermarked_signal(PAM_frame));
    shaping_filter_response = masking_threshold;
    [b0, ai] = shaping_filter_design(shaping_filter_response, N_coef);
    
    % Filtering stage
    [equalized_signal(PAM_frame), state] = ...
        filter(ai./b0, 1, watermarked_signal(PAM_frame), state);

    if m==10
    % Showing the frequency response of the equalizer and PSD of the
    % |equalized_signal|, for frame 10.
        close(1)
        pwelch(equalized_signal(PAM_frame),[],[],[],2);
        hold on;
        [H,W]=freqz(ai./b0,1,256);
        plot(W/pi,20*log10(abs(H)),'r','linewidth',2);
        hold off;
        xlabel('Normalized Frequency : f/(Fs/2)');
        legend('Equalized signal', 'Equalizer response', 1);
    end;

end

%%
% The |equalized_signal| is theoretically the sum of the original watermark
% (the |modulated_signal|, which has values in {-1,+1}) and the
% |equalized_audio_signal|, which is itself the original |audio_signal|
% filtered by the Zero-Forcing equalizer. In other words, from the receiver
% point of view, everything looks as if the watermark had been added to the
% |equalized_audio_signal| rather than to the |audio_signal| itself. 
% Although the |equalized_audio_signal| is not available to the receiver, it
% is interesting to compute it and compare it to the original |audio_signal|.

equalized_audio_signal = zeros(N_bits*N_samples, 1);  
state = zeros(N_coef, 1);

for m = 0:fix(N_bits*N_samples/N_samples_PAM)-1
    
    PAM_frame = m*N_samples_PAM+1:m*N_samples_PAM+N_samples_PAM;
    
    % Shaping filter design, based the watermarked signal
    masking_threshold = psychoacoustical_model(watermarked_signal(PAM_frame));
    shaping_filter_response = masking_threshold;
    [b0, ai] = shaping_filter_design(shaping_filter_response, N_coef);
    
    % Filtering stage
    [equalized_audio_signal(PAM_frame), state] = ...
        filter(ai./b0, 1, audio_signal(PAM_frame), state);
end

% Plotting 200 samples of the |equalized_audio_signal| and
% |modulated_signal|. Notice the level of the |equalized_audio_signal| is
% much higher than that of the original |audio_signal|, mostly because its
% HF content has been enhanced by the equalizer. 

clf;
range=(34739:34938);
subplot(2,1,1); plot(range/Fs, equalized_audio_signal(range));
set(gca,'xlim',[range(1)/Fs range(end)/Fs]);
subplot(2,1,2); plot(range/Fs, modulated_signal(range));
set(gca,'xlim',[range(1)/Fs range(end)/Fs],'ylim',[-1.1,1.1]);
xlabel('Time(s)');

%%
% The SNR can be estimated from the PSDs of the |equalized_audio_signal|
% and |modulated_signal|, or computed easily from the samples of these
% signals 

close(1)
range=(34398:34938);
pwelch(equalized_audio_signal(range),[],[],[],2);
[h,w]=pwelch(modulated_signal(range),[],[],[],2);
xlabel('Normalized Frequency : f/(Fs/2)');
hold on;
plot(w,10*log10(h),'--r ', 'linewidth',2);
set(gca,'ylim',[-30 50]);
legend('noise','signal');

snr_equalized=10*log10(var(modulated_signal(range))/var(equalized_audio_signal(range)))

%%
% We can now proceed to the watermark extraction by applying the
% correlation demodulator to the |equalized_signal| and compute
% the BER. As expected, the obtained BER is quite high, since the level of
% the |equalized_audio_signal| is high compared to that of the embedded
% |modulated_signal| (in other words, the SNR is low). 

for m = 0:N_bits-1
    alpha(m+1) = (equalized_signal(m*N_samples+1: ...
        m*N_samples+N_samples)' * spread_waveform)/N_samples;
    if alpha(m+1) <= 0
        received_bits(m+1) = 0;
    else
        received_bits(m+1) = 1;
    end
end

% Plotting the input bits, the value of alpha, and the received bits.
clf;
range=(81:130);
subplot(3,1,1); 
stairs(range, bits(range));
set(gca,'ylim',[-0.1 1.1]);
subplot(3,1,2); 
stairs(range, alpha(range));
subplot(3,1,3); 
stairs(range, received_bits(range));
set(gca,'ylim',[-0.1 1.1]);
xlabel('frame index');

number_of_erroneous_bits= sum(bits ~= received_bits)
total_number_of_bits=N_bits
BER = number_of_erroneous_bits/total_number_of_bits

received_chars = reshape(received_bits(1:N_bits),N_bits/7, 7);
received_message = char( bin2dec(num2str(received_chars)))'

%% 3.3 Wiener filtering
% The Wiener filtering stage aims at enhancing the SNR between the 
% |modulated_signal| and the |equalized_audio_signal|.
% This is achieved here by filtering the |equalized_signal| by a
% symmetric (non causal) FIR filter with |N_coef|=50 coefficients. 
% Its coefficients |hi| are computed so that the output of the filter (when
% fed with the |equalized_signal|) becomes maximally similar to the
% modulated_signal in the RMSE sense. They are the solution of the
% so-called Wiener-Hopf equations:
%
% |hi = inv(equalized_signal_cov_mat)*modulated_signal_autocor_vect|
%
% where |equalized_signal_cov_mat| and |modulated_signal_autocor_vect| are
% the covariance matrix of the |equalized_signal| and the autocorrelation
% vector of the |modulated_signal|, respectively. 
%
% Since the |modulated_signal| is unknown from the receiver, its
% autocorrelation is estimated from an arbitrary modulated signal and can
% be computed once. To make it simple, we use our previously computed
% |modulated_signal| here. On the contrary, the covariance matrix of the
% equalized signal, and therefore the coefficients |hi|, have to be updated
% each time the properties of |estimated_signal| change,
% that is every |N_samples_PAM|=512 samples.
% Wiener filtering is then carried out, by computing the
% covariance matrix of the |equalized_signal| and the impulse response of
% the Wiener filter for each PAM frame. 

modulated_signal_autocor_vect= xcorr(modulated_signal,N_coef,'biased');

Wiener_output_signal = zeros(N_bits*N_samples + N_coef,1);        
state = zeros(2*N_coef, 1);

for m = 0:fix(N_bits*N_samples/N_samples_PAM)-1
    PAM_frame = (m*N_samples_PAM+1:m*N_samples_PAM+N_samples_PAM);
    
    % Estimating the covariance matrix of |equalized signal|, as a Toeplitz
    % matrix with first row given by the autocorrelation vector of the
    % signal.
    equalized_signal_autocor_vect=xcorr(equalized_signal(PAM_frame),2*N_coef,'biased');
    equalized_signal_cov_mat= toeplitz(equalized_signal_autocor_vect(2*N_coef+1:end));
    
    % Estimating the impulse response of the Wiener filter as the solution
    % of the Wiener-Hopf equations.
    hi=equalized_signal_cov_mat\modulated_signal_autocor_vect;
    
    % Filtering stage
    [Wiener_output_signal(PAM_frame,1), state] = ...
        filter(hi, 1, equalized_signal(PAM_frame), state);
    power = norm(Wiener_output_signal(PAM_frame))^2/N_samples_PAM;
    if (power ~= 0)
        Wiener_output_signal(PAM_frame) = Wiener_output_signal(PAM_frame)/ sqrt(power);
    end
    
    % Saving the Wiener filter for frame #78
    if m==78
        hi_78=hi/sqrt(power);
    end;
end

% Filtering the last, incomplete frame in |equalized_signal|
PAM_frame = (m*N_samples_PAM+N_samples_PAM+1:N_bits*N_samples);
Wiener_output_signal(PAM_frame,1) = ...
    filter(hi, 1, equalized_signal(PAM_frame), state);
power = norm(Wiener_output_signal(PAM_frame))^2/N_samples_PAM;
if (power ~= 0)
    Wiener_output_signal(PAM_frame)= Wiener_output_signal(PAM_frame) /sqrt(power);
end

% Since the Wiener filter is non-causal (with |N_coeff|=50
% coefficients for the non-causal part), the resulting
% |Wiener_output_signal| is delayed with |N_coef| samples. 
Wiener_output_signal = [Wiener_output_signal(N_coef+1:end);  zeros(N_coef, 1)];

%%
% It is interesting to check how the equalized audio signal and the
% modulated signal have been modified by the Wiener filter.

clf;
range=(34398:34938);
Wiener_output_audio=filter(hi_78,1,equalized_audio_signal(range));
Wiener_output_modulated=filter(hi_78,1,modulated_signal(range));
range=(34739:34938);
subplot(2,1,1); plot(range/Fs, Wiener_output_audio(341:540));
set(gca,'xlim',[range(1)/Fs range(end)/Fs]);
subplot(2,1,2); plot(range/Fs, Wiener_output_modulated(341:540));
set(gca,'xlim',[range(1)/Fs range(end)/Fs]);
xlabel('Time(s)');

%%
% The new SNR can be estimated from the PSDs of the filtered
% |equalized_audio_signal| and |modulated_signal|. Obviously, the Wiener
% filter has enhanced frequency bands dominated by the modulated signal.
% increased, thereby increasing the SNR.

close(1)
pwelch(Wiener_output_audio,[],[],[],2);
[h,w]=pwelch(Wiener_output_modulated,[],[],[],2);
xlabel('Normalized Frequency : f/(Fs/2)');
hold on;
plot(w,10*log10(h),'--r ', 'linewidth',2);
[h,w]=freqz(hi_78,1);
plot(w/pi,20*log10(h),':k ', 'linewidth',2);
set(gca,'ylim',[-60 30]);
legend('noise','signal','Wiener filter response');

snr_Wiener=10*log10(var(Wiener_output_modulated(341:530)) ...
    /var(Wiener_output_audio(341:530)))

%%
% We can finally apply the correlation demodulator to the estimated
% modulated signal. The resulting BER is higher thanks to Wiener filtering. 
% It has the same order of magnitude as the one we obtained in Section 1,
% while the watermark is now inaudible.

for m = 0:N_bits-1
    alpha(m+1) = (Wiener_output_signal(m*N_samples+1: ...
        m*N_samples+N_samples)'*spread_waveform )/N_samples;
    if alpha(m+1) <= 0
        received_bits(m+1) = 0;
    else
        received_bits(m+1) = 1;
    end
end

% Plotting the input bits, the value of aplha, and the received bits.
clf;
range=(81:130);
subplot(3,1,1); 
stairs(range, bits(range));
set(gca,'ylim',[-0.1 1.1]);
subplot(3,1,2); 
stairs(range, alpha(range));
subplot(3,1,3); 
stairs(range, received_bits(range));
set(gca,'ylim',[-0.1 1.1]);
xlabel('frame index');

number_of_erroneous_bits= sum(bits ~= received_bits)
total_number_of_bits=N_bits
BER = number_of_erroneous_bits/total_number_of_bits

received_chars = reshape(received_bits(1:N_bits),N_bits/7, 7);
received_message = char( bin2dec(num2str(received_chars)))'

%% 3.4 Robustness to MPEG compression
% We finally focus on the robustness of our system to MPEG compression.
% Using the MATLAB functions developed in Chapter 3, we can easily apply
% an mp3 coding/decoding operation to the audio watermarked signal,
% yielding the distorted audio watermarked signal
% |compressed_watermarked_signal|. 
%
% *MATLAB function involved:*
%
% * |output_signal = codec_mp3( input_signal, Fs)| returns the signal
% |output_signal| resulting from a mp3 coding/decoding operation of the
% input signal |input_signal| sampled at the |Fs| frequency sampling. (See
% chapter 3)

compressed_watermarked_signal = mp3_codec( watermarked_signal, Fs );

%%
% We then pass the compressed signal through the Zero-Forcing equalizer and
% the Wiener filter

modulated_signal_autocor_vect= xcorr(modulated_signal,N_coef,'biased');
state1 = zeros(N_coef, 1);
state3 = zeros(2*N_coef, 1);
for m = 0:fix(N_bits*N_samples/N_samples_PAM)-1
    PAM_frame = (m*N_samples_PAM+1:m*N_samples_PAM+N_samples_PAM);

    % Zero-Forcing filtering
    masking_threshold = psychoacoustical_model( compressed_watermarked_signal(PAM_frame) );
    shaping_filter_response = masking_threshold;
    [b0, ai] = shaping_filter_design(shaping_filter_response, N_coef);
    [filtered_signal(PAM_frame,1), state1] = ...
        filter(ai./b0, 1, compressed_watermarked_signal(PAM_frame), state1);

    % Wiener filtering
    filtered_signal_autocor_vect=xcorr(filtered_signal(PAM_frame),2*N_coef,'biased');
    filtered_signal_cov_mat= toeplitz(filtered_signal_autocor_vect(2*N_coef+1:end));
    hi=filtered_signal_cov_mat\modulated_signal_autocor_vect;
    [Wiener_filtered_signal(PAM_frame,1), state3] = ...
        filter(hi, 1, filtered_signal(PAM_frame), state3);
    power = norm(Wiener_filtered_signal(PAM_frame))^2/N_samples_PAM;
    if (power ~= 0)
        Wiener_filtered_signal(PAM_frame) = Wiener_filtered_signal(PAM_frame)/sqrt(power);
    end
end
% Filtering the last, incomplete frame in |filtered_signal|
PAM_frame = (m*N_samples_PAM+N_samples_PAM+1:N_bits*N_samples);
filtered_signal(PAM_frame,1) = ...
    filter(ai./b0, 1, compressed_watermarked_signal(PAM_frame), state1);
Wiener_filtered_signal(PAM_frame,1) = ...
    filter(hi, 1, filtered_signal(PAM_frame), state3);

% Wiener delay suppression
Wiener_filtered_signal =  [Wiener_filtered_signal(N_coef+1:N_bits*N_samples); zeros(N_coef, 1)];

%%
% We finally apply our correlation detection scheme to this distored signal
% and compute the BER. 

for m = 0:N_bits-1
    alpha(m+1) = Wiener_filtered_signal(m*N_samples+1:m*N_samples+N_samples)'* ...
        spread_waveform/N_samples;
    if alpha(m+1) <= 0
        received_bits(m+1) = 0;
    else
        received_bits(m+1) = 1;
    end
end
number_of_erroneous_bits= sum(bits ~= received_bits)
total_number_of_bits=N_bits
BER = number_of_erroneous_bits/N_bits

received_chars = reshape(received_bits(1:N_bits),N_bits/7, 7);
received_message = char( bin2dec(num2str(received_chars)))'


%%
% Unfortunately, the obtained BER has significantly increased: although it
% has been shown above that the psychoacoustic model we use in our
% perceptual watermarking system is robust to the watermark (i.e., the
% masking threshold does not change significantly when the watermark is
% added to the audio signal), it is clearly  not robust yet to MPEG
% compression.  
%
% To show this, let us compare the PSD of the watermarked signal and the
% associated masking threshold, before and after the MPEG compression, on
% the 11th frame for instance. Clearly, the masking threshold is very
% sensitive to MPEG compression for frequencies over 11 kHz. As a result,
% the perceptual shaping filter and the equalizer no longer cancel each
% other, hence our high BER.

PAM_frame = (10*N_samples_PAM+1 : 10*N_samples_PAM+N_samples_PAM);

% Plotting the masking threshold, before MPEG compression
close(1); 
subplot(1,2,1)
pwelch(watermarked_signal(PAM_frame),[],[],[],2);
hold on;
masking_threshold = psychoacoustical_model( watermarked_signal(PAM_frame) );
stairs((1:N_samples_PAM/2)*2/N_samples_PAM,masking_threshold,'r','linewidth',2);
xlabel('Normalized Frequency : f/(Fs/2)');
legend('Audio signal', 'Masking threshold', 1);
axis([0 1 -100 0]);

% Plotting the masking threshold, after MPEG compression
subplot(1,2,2)
pwelch(compressed_watermarked_signal(PAM_frame),[],[],[],2);
hold on;
masking_threshold = psychoacoustical_model( compressed_watermarked_signal(PAM_frame) );
stairs((1:N_samples_PAM/2)*2/N_samples_PAM,masking_threshold,'r','linewidth',2);
xlabel('Normalized Frequency : f/(Fs/2)');
legend('Audio signal', 'Masking threshold', 1);
axis([0 1 -100 0]);

%% 4. Informed watermarking, robust to MPEG compression
% In order to improve the robustness of the system exposed in the previous
% Section, the watermark information should obviously be spread in the [0
% Hz, 11 kHz] frequency range of the audio signal. This can be achieved
% with a low pass filter with cutoff frequency set to 11 kHz. As we shall
% see below, this filter will interfere with our watermarking system in
% three stages.

% Preparing the data, as in Section 2.
message = 'Audio watermarking: this message will be embedded in an audio signal, using spread spectrum modulation, and later retrieved from the modulated signal.'
bits = dec2bin(message); % converting each ascii character to a 7-bit string
bits = bits(:)'; % reshaping to a single string of 1's and 0's
bits = double(bits) - 48; % converting to a (1x1050) array of 0's and 1's
symbols = bits*2-1;  % a (1x1050) array of -1's and +1's
N_bits = length(bits);
rand('seed', 12345);
randn('seed', 12345);
Fs = 44100;
R = 100;
N_samples = round(Fs/R);
SNR_dB = -20;
spread_waveform = 2*round( rand(N_samples, 1) ) - 1; % in {-1,1}
modulated_signal = zeros(N_bits*N_samples, 1);
for m = 0:N_bits-1
    modulated_signal(m*N_samples+1:m*N_samples+N_samples) = ...
        symbols(m+1)*spread_waveform;
end
audio_signal = wavread('violin.wav', [1 N_bits*N_samples]);

%% 4.1 Designing the low pass filter
% First, we design a symmetric FIR low pass filter with |Fc=11| kHz cutoff
% frequency, using the Parks-McClellan algorithm. Such a filter will have
% linear phase, and therefore will not change the shape of the modulated
% signal more than required.
% We set the order of the filter to |N_coef=50|. Notice that this
% filter is non-causal and introduces a |N_coef/2|-samples delay.

Fc = 11000;
low_pass_filter = firpm(N_coef, [0 Fc-1000 Fc+1000 Fs/2]*2/Fs, [1 1 1E-9 1E-9]);
%low_pass_filter = low_pass_filter/sqrt(sum(low_pass_filter)^2));

% Let us plot the impulse response of this filter.
clf;
plot(low_pass_filter);

%%
% It is easy to check that its frequency response matches our requirements. 
clf;
freqz(low_pass_filter,1,256,Fs);

%% 4.2 Modifying the emitter
% We first use this low-pass filter to create a spread sequence with
% spectral content restricted to the [0 Hz, 11 kHz] range.  The modulated  
% signal can then be designed as previously. 

spread_waveform = filter(low_pass_filter, 1, 2*round(rand(N_samples+N_coef, 1)) - 1);
% Getting rid of the transient
spread_waveform = spread_waveform(N_coef+1:end); 
% Making sure the norm of the spread waveform is set to sqrt(N_samples), as
% in the previous Sections.
spread_waveform = spread_waveform/ ...
    sqrt(norm(spread_waveform).^2/N_samples);

for m = 0:N_bits-1
    modulated_signal(m*N_samples+1:m*N_samples+N_samples) = ...
        symbols(m+1)*spread_waveform;
end

%% 
% To prevent the shaping filter from amplifying the residual
% frequency component of the modulated signal over 11 kHz, the
% |shaping_filter_response|, initially chosen to be equal to the masking
% threshold, is now designed to be equal to the masking threshold in the [0, 11] kHz
% frequency band and to zero anywhere else. 

state = zeros(N_coef, 1);
N_cutoff = ceil(Fc/(Fs/2)*(N_samples_PAM/2));

for m = 0:fix(N_bits*N_samples/N_samples_PAM)-1
    PAM_frame = (m*N_samples_PAM+1:m*N_samples_PAM+N_samples_PAM);

    % Shaping filter design
    masking_threshold = psychoacoustical_model( audio_signal(PAM_frame) );
    shaping_filter_response = [masking_threshold(1:N_cutoff); ...
        -100*ones(N_samples_PAM/2-N_cutoff, 1) ];
    [b0, ai] = shaping_filter_design(shaping_filter_response, N_coef);
    
    % Filtering stage
    [watermark_signal(PAM_frame), state] = ...
        filter(b0, ai, modulated_signal(PAM_frame), state);
end
% Filtering the last, incomplete frame in |watermark_signal|
PAM_frame = (m*N_samples_PAM+N_samples_PAM+1:N_bits*N_samples);
watermark_signal(PAM_frame,1) = ...
    filter(b0, ai, modulated_signal(PAM_frame), state);

watermarked_signal = audio_signal + watermark_signal;

%%
% MPEG compression is then applied to the new |watermarked_signal|.

compressed_watermarked_signal = mp3_codec( watermarked_signal, Fs );

%% 4.3 Modifying the receiver
% Before starting the watermark extraction, the
% |compressed_watermarked_signal| is low-passed, to avoid residual 
% components over 11 kHz.

compressed_watermarked_signal = ...
    filter(low_pass_filter, 1, [compressed_watermarked_signal; zeros(N_coef/2, 1)]);
% Compensating tof the |N_coef/2| delay
compressed_watermarked_signal = ...
    compressed_watermarked_signal(N_coef/2+1:end);

%%
% Watermark extraction is finally obtained as in Section 3, except the LP
% filter if still taken into account in the zero-forcing equalization. 
% The resulting BER is now similar to the one we obtained with no MPEG compression.
% The watermarking system is thus now robust to MPEG compression.

modulated_signal_autocor_vect= xcorr(modulated_signal,N_coef,'biased');

for m = 0:fix(N_bits*N_samples/N_samples_PAM)-1
    PAM_frame = (m*N_samples_PAM+1:m*N_samples_PAM+N_samples_PAM);

    % Zero-Forcing filtering
    masking_threshold = psychoacoustical_model(compressed_watermarked_signal(PAM_frame));
    shaping_filter_response = [masking_threshold(1:N_cutoff); ...
        -100*ones(N_samples_PAM/2-N_cutoff, 1) ];
    [b0, ai] = shaping_filter_design(shaping_filter_response, N_coef);

    [filtered_signal(PAM_frame), state1] = ...
        filter(ai./b0, 1, compressed_watermarked_signal(PAM_frame), state1);

    % Wiener egalization
    filtered_signal_autocor_vect=xcorr(filtered_signal(PAM_frame),2*N_coef,'biased');
    filtered_signal_cov_mat= toeplitz(filtered_signal_autocor_vect(2*N_coef+1:end));
    hi=filtered_signal_cov_mat\modulated_signal_autocor_vect;
    
    [Wiener_filtered_signal(PAM_frame), state3] = ...
        filter(hi, 1, filtered_signal(PAM_frame), state3);
    power = norm(Wiener_filtered_signal(PAM_frame))^2/N_samples_PAM;
    if (power ~= 0)
        Wiener_filtered_signal(PAM_frame) = Wiener_filtered_signal(PAM_frame)/ sqrt(power);
    end
end
% Filtering the last, incomplete frame
PAM_frame = (m*N_samples_PAM+N_samples_PAM+1:N_bits*N_samples);
Wiener_filtered_signal(PAM_frame) = filter(hi, 1, filtered_signal(PAM_frame), state3);
power = norm(Wiener_filtered_signal(PAM_frame))^2/N_samples_PAM;
if (power ~= 0)
    Wiener_filtered_signal(PAM_frame)= Wiener_filtered_signal(PAM_frame) /sqrt(power);
end
Wiener_filtered_signal = [Wiener_filtered_signal(N_coef+1:N_bits*N_samples); zeros(N_coef, 1)];

% Correlation demodulator
for m = 0:N_bits-1
    alpha(m+1) =  Wiener_filtered_signal(m*N_samples+1:m*N_samples+N_samples)'* ...
        spread_waveform /N_bits;
    if alpha(m+1) <= 0
        received_bits(m+1) = 0;
    else
        received_bits(m+1) = 1;
    end
end
number_of_erroneous_bits= sum(bits ~= received_bits)
total_number_of_bits=N_bits
BER = number_of_erroneous_bits/N_bits

received_chars = reshape(received_bits(1:N_bits),N_bits/7, 7);
received_message = char( bin2dec(num2str(received_chars)))'
