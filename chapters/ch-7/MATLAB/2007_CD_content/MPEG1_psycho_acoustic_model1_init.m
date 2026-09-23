% Constants used in the MPEG-1 psycho-acoustic model #1
% (for Fs=44100 Hz)
%
% Copyright N. Moreau, ENST Paris, 19/3/02

global LTq_i LTq_k Table_z Frontieres_i Frontieres_k Larg_f

% "i" to "k"
i1 = 1:48;
i2 = (49:72)-48;
i3 = (73:108)-72;
i_to_k = [i1 (2*i2+48) (4*i3+96)];

% Absolute auditory threshold as a function of "i"
LTq_i = [25.87 14.85 10.72  8.50  7.10  6.11  5.37  4.79 ...
             4.32  3.92  3.57  3.25  2.95  2.67  2.39  2.11 ...
             1.83  1.53  1.23  0.90  0.56  0.21 -0.17 -0.56 ...
            -0.96 -1.38 -1.79 -2.21 -2.63 -3.03 -3.41 -3.77 ...
            -4.09 -4.37 -4.60 -4.78 -4.91 -4.97 -4.98 -4.92 ...
            -4.81 -4.65 -4.43 -4.17 -3.87 -3.54 -3.19 -2.82 ...
            -2.06 -1.32 -0.64 -0.04  0.47  0.89  1.23  1.51 ...
             1.74  1.93  2.11  2.28  2.46  2.63  2.82  3.03 ...
             3.25  3.49  3.74  4.02  4.32  4.64  4.98  5.35 ...
             6.15  7.07  8.10  9.25 10.54 11.97 13.56 15.31 ...
            17.23 19.34 21.64 24.15 26.88 29.84 33.05 36.52 ...
            40.25 44.27 48.59 53.22 58.18 63.49 68.00 68.00 ...
            68.00 68.00 68.00 68.00 68.00 68.00 68.00 68.00 ...
            68.00 68.00 68.00 68.00];

% Absolute auditory threshold as a function of "k"
LTq_k = zeros(250, 1);
for i = 1:length(LTq_i)
    LTq_k(i_to_k(i)) = LTq_i(i);
end
for k = 1:250
    if LTq_k(k) == 0
        LTq_k(k) = last_nonzero;
    else
        last_nonzero = LTq_k(k);
    end
end

% axe_freq = (0:249)*Fe/512/1000;
% figure(1); hold off
% plot(axe_freq, LTq_k); hold on
% axis([0 20 -10 70])

% Bark frequencies as a fucntion of "i"
Table_z = [ .850 1.694 2.525 3.337 4.124 4.882 5.608 6.301 ...
             6.959  7.581  8.169  8.723  9.244  9.734 10.195 10.629 ...
            11.037 11.421 11.783 12.125 12.448 12.753 13.042 13.317 ...
            13.578 13.826 14.062 14.288 14.504 14.711 14.909 15.100 ...
            15.284 15.460 15.631 15.796 15.955 16.110 16.260 16.406 ...
            16.547 16.685 16.820 16.951 17.079 17.205 17.327 17.447 ...
            17.680 17.905 18.121 18.331 18.534 18.731 18.922 19.108 ...
            19.289 19.464 19.635 19.801 19.963 20.120 20.273 20.421 ...
            20.565 20.705 20.840 20.972 21.099 21.222 21.342 21.457 ...
            21.677 21.882 22.074 22.253 22.420 22.576 22.721 22.857 ...
            22.984 23.102 23.213 23.317 23.415 23.506 23.592 23.673 ...
            23.749 23.821 23.888 23.952 24.013 24.070 24.125 24.176 ...
            24.225 24.271 24.316 24.358 24.398 24.436 24.473 24.508 ...
            24.542 24.574 25 25];
    
Frontieres_i = [1 2 3 5 6 8 9 11 13 15 17 20 23 27 32 37 ...
          45 50 55 61 68 75 81 93 106];

Frontieres_k = zeros(1, length(Frontieres_i));
for i = 1:length(Frontieres_i)
    Frontieres_k(i) = i_to_k(Frontieres_i(i));
end

% Bandwidth of critical bands
f_250_c = [0 Frontieres_k 296];
f_250_d = [0 Frontieres_k 256];

upper_bound = 1;
lower_bound = 1;
while upper_bound < 250
    lower_bound = lower_bound + 1;
    for k = 1:25
        if lower_bound >= f_250_c(k) & lower_bound < f_250_c(k+1)
            larg_bas = f_250_c(k+1) - f_250_c(k);
            no_ech_bas = f_250_c(k+1) - lower_bound;
        end
    end
    if no_ech_bas >= ceil(larg_bas/2)
        larg_fen = ceil(larg_bas/2);
    else
        for k = 1:25
            if upper_bound >= f_250_c(k) & upper_bound < f_250_c(k+1)
                larg_haut = f_250_c(k+1) - f_250_c(k);
                no_ech_haut = upper_bound - f_250_c(k);
            end
        end
        no_ech_tot = no_ech_haut + no_ech_bas;
        larg_fen = ceil((larg_bas*no_ech_bas/no_ech_tot+larg_haut*no_ech_haut/no_ech_tot)/2);
    end
    upper_bound = lower_bound + larg_fen;
    Larg_f(lower_bound) = larg_fen;
end
