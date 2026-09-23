function [snr_value, shift] = snr(signal,signal_plus_noise,max_shift,showplot)

% [snr_value, shift] = snr(signal,signal_plus_noise, max_shift,showplot) returns the
% signal-to-noise ratio computed from the input signals. |Max_shift| gives
% the maximum time-shift (in samples) between signal and signal_plus_noise.
% The actual time-shift (obtained from the maximum of the cross-correlation
% and returned as |shift|) is taken into account to estimate the noise. If
% signal are of different length, the shortest length is used.
% If |showplot| is specified, then the signal, signal_plus_noise, and error 
% are plotted, and the SNR is printed on the plot.
% 
% T DUTOIT, 13:49 12/03/2007


sig_length=min(length(signal),length(signal_plus_noise));
sig_length=floor(sig_length/2)*2; % make it even
signal=signal(1:sig_length); 
signal_plus_noise=signal_plus_noise(1:sig_length);

len=min([max(10*max_shift,1000) sig_length-1]);
half_len=floor(len/2);
center=floor(sig_length/2);
cross_correlation = xcorr(signal_plus_noise(center-half_len:center+half_len-1),...
   signal(center-half_len:center+half_len-1),max_shift,'biased');
[max_value,max_lag] = max(cross_correlation(max_shift+1:2*max_shift+1));
shift=max_lag-1;

tmp=signal_plus_noise(1+shift:length(signal_plus_noise));
noise=tmp-signal(1:length(tmp));

% Uncomment this to see the signals on which the SNR is computed
% plot(signal(1:length(tmp))); hold on;plot(tmp,'r'); plot(noise,'g'); hold off;

var_signal_dB=10*log10(var(signal));
var_noise_dB=10*log10(var(noise));
snr_value=var_signal_dB-var_noise_dB;

if nargin==4
    plot(signal(center-half_len:center+half_len-1))
    hold on;
	plot(tmp(center-half_len:center+half_len-1),'r')
	plot(noise(center-half_len:center+half_len-1),'g')
    legend('signal','signal+noise','noise');
    title(['SNR = ' num2str(snr_value)]);
    hold off;
end;
