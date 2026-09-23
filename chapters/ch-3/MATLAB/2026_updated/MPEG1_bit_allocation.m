function [N_bits,SNR] = MPEG1_bit_allocation1(SMR, bit_rate)

% |function [N_bits,SNR] = MPEG1_bit_allocation(SMR, bit_rate)|
% Implements a simplified bit allocation greedy algorithm.
% |SMR| is the signal-to-mask ratios in each sub-band,
% as defined by the MPEG1 psycho-acoustic model. 
% |bit_rate| is in kbits/s.
% |N_bits| is the number is bits in each sub_band.
% |SNR| is the maximum SNR in each sub-band after quantization, i.e. the
% SNR assuming each sub-band contains a full-range sinusoid.
% NB: N_bits and SNR are set to zero for sub-bands 28 to 32.
% 
% Copyright N. Moreau, ENST Paris, 19/03/02
% Modified by Thierry Dutoit, FPMs Mons, 03/05/07
 
Fe=44100;
% Imposing low SMR for bands 28-32, to ensure 0 bits.
SMR = [SMR(1:27) ones(1,5)*-100]; 

% Assuming 6 bits for each scale factor, and 4 bits for each number of
% allocated bits per sub-band
available_bits_per_block = bit_rate*384/Fe- 6*27-4*27;
used_bits=0;

% Mask-to-Noise Ratio, should be maximized.
% Initialised assuming no bit used in any frame (SNR=0)
N_bits = zeros(1,32);
SNR = zeros(1,32);
MNR = SNR-SMR;

iter = 0;
iter_max = 1000;
while used_bits < available_bits_per_block
	[temp kmin] = min(MNR);
    if N_bits(kmin) == 16
		SNR(kmin) = 100; % avoid more then 16 bits
	else
		if N_bits(kmin)==0
            % Avoid having 1 bit only
            N_bits(kmin) = 2;
        else
            N_bits(kmin) = N_bits(kmin) + 1;
        end
        % Assuming sub-band signal = full scale cosine
        SNR(kmin)=1.77+6.02*N_bits(kmin);
    end
    MNR(kmin) = SNR(kmin) - SMR(kmin);
	used_bits = 12*sum(N_bits);
	iter = iter + 1;
    if iter > iter_max
        % More bits than needed
        return
    end
end