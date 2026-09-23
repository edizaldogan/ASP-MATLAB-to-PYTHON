function [b0, ai] = shaping_filter_design(desired_frequency_response_dB, N_coef)

% * |[b0, ai] = shaping_filter_design(desired_frequency_response_dB, N_coef)| 
% computes the coefficients of the auto-regressive filter:  
%
%                        b0
% G(z) = ---------------------------------------------
%                      -1                    -N_coef+1
%        ai(1) + ai(2)z   + ... + ai(N_coef)z
%
% with ai(1) = 1, from the modulus of its |desired_frequency_response| 
% (in dB) and the order |N_coef|. 
%
% The coefficients are obtained as follows: if zero-mean and unity variance
% noise is provided at the input of the filter, the PSD of its ouput is
% given by |desired_frequency_response|. Setting the coefficients so that 
% this PSD best matches |desired_frequency_response| is thus obtained by
% applying the Levinson algorithm to the autocorrelation coefficients of
% the output signal (computed itslef from the IFFT of the
% |desired_frequency_response|). 

N = 2*length(desired_frequency_response_dB);

% Autocorrelation coefficients
filter_response = 10.^(desired_frequency_response_dB/10); 
rk = ifft( [filter_response; filter_response(N/2); flipud(filter_response(2:N/2))] );
rk = real(rk);

% Levinson's procedure
ai = zeros(N_coef+1, 1);
[ai, error] = levinson( rk(1:N_coef+1), N_coef );
ai = ai';
b0 = sqrt(error);


