function masking_threshold = psychoacoustical_model( audio_signal )

% * |masking_threshold = psychoacoustical_model( audio_signal )| returns
% the masking threshold deduced from a psychoacoustical analysis of
% the audio vector. 
%
% This implementation is derived from the psycho-acoustic model #1 used in
% MPEG-1 Audio (see ISO/CEI norm 11172-3:1993 (F), pp. 122-128 or Matlab
% function MPEG1_psycho_acoustic_model1.m from Chapter 3). It is based on
% the same principles as those used in the MPEG model, but it is further
% adapted here so as to make it robust to additive noise (which is a
% specific constraint of watermarking and is not found in MPEG).

N = length(audio_signal);

% Power Spectral Density (PSD) computation
audio_fft = fft( audio_signal.*hanning(N) ); 
audio_PSD = 10*log10( abs(audio_fft(1:N/2)).^2 /N );

% Dynamic compression of the PSD over 4 "frequency critical bands" 
Nk = N/8;
compressed_PSD = zeros(N/2, 1);
for k = 0:3
    mk = mean( audio_PSD(k*Nk+1:k*Nk+Nk) );
    compressed_PSD( k*Nk+1:k*Nk+Nk ) = ( audio_PSD(k*Nk+1:k*Nk+Nk)-mk )/2 + mk;
end

% Spreading function application
P = 10;
spreading_function = [-0.0052 -0.008 0.0134 0.1057 0.2405 0.3072 0.2405 0.1057 ...
        0.0134 -0.008 -0.0052]';
    
input_PSD = [ones(P/2,1)*compressed_PSD(1); compressed_PSD; ...
        ones(P/2,1)*compressed_PSD(N/2)];  
filtered_PSD = filter(spreading_function, 1, input_PSD);
filtered_PSD = filtered_PSD(P+1:P+N/2);

% Smoothing over the 3 "high frequency critical bands"
masking_threshold = filtered_PSD;

% 2d critical band
for i = Nk+1:2:2*Nk
    masking_threshold(i:i+1) = mean( filtered_PSD(i:i+1) )*ones(2, 1);
end

% 3d/4th critical band
for i = 2*Nk+1:3:4*Nk-2
    masking_threshold(i:i+2) = mean( filtered_PSD(i:i+2) )*ones(3, 1);
end
masking_threshold(N/2-1:N/2) = masking_threshold(N/2-2)*ones(2, 1);

% Tuning : this value can be changed, so as to tune the inaudibility contraint 
masking_threshold=masking_threshold-5;