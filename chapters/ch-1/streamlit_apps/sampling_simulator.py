
import streamlit            as st      #always first line
import numpy                as np
import matplotlib.pyplot    as plt
#import scipy.signal         as signal


st.set_page_config(page_title="Sampling Simulator",
                   page_icon="🦈",layout="wide",
                   initial_sidebar_state="collapsed")
st.title('Sampling Simulator')

# User sets the following:

input_freq      = st.slider('Input frequency [Hz]'   , 0, 10, 1, 1)
sampling_freq   = st.slider('Sampling Frequency [Hz]', 1, 100, 37, 1)
phi             = st.slider('Phase phi [rad]'        , -np.pi, np.pi)

freq_zoom       = st.slider('Frequency zoom'         , 1, 100, 100)
time_zoom       = st.slider('Time zoom'              , 1, 100, 100)

time_interval_to_be_shown = time_zoom/100
frequency_interval_to_be_shown = freq_zoom


simulation_freq = 48000
t_of_simulation = np.arange(0, 1, 1/simulation_freq)
T_sampling      = 1/sampling_freq
t_of_sampling   = np.arange(0, 1+T_sampling, T_sampling)


# LEFT side signals in time domain
input_signal         = np.sin(2*np.pi*input_freq*t_of_simulation + phi)
impulse_locations    = np.arange(0,1+T_sampling,T_sampling)
impulse_train        = np.ones_like(impulse_locations)
sampled_signal       = np.sin(2*np.pi*input_freq*impulse_locations + phi)



K = int(np.round(simulation_freq / sampling_freq))

impulse_train_extended      = np.zeros_like(t_of_simulation)
#put a 1 in every period (impulse locations)
impulse_train_extended[::K] = 1

sampled_signal_extended      = np.zeros_like(t_of_simulation)
sampled_signal_extended[::K] = input_signal[::K]




staircase_sampled_signal = np.repeat(sampled_signal, K)
# staircase_sampled_signal has more than 48k points.
print(staircase_sampled_signal, len(staircase_sampled_signal))

reconstructed_signal = staircase_sampled_signal[:48000]
print('recosntructed signal is:', reconstructed_signal, len(reconstructed_signal))


# RIGHT side graphs in frequency domain
#after sampling, many of the info is lost, we ragain the power by multiplying with 
# K which is sim_fre/sampl_freq
input_signal_spectrum            = np.fft.fft(input_signal)/len(t_of_simulation)
impulse_train_spectrum           = np.fft.fft(impulse_train_extended)*K/len(t_of_simulation)
sampled_input_signal_spectrum    = np.fft.fft(sampled_signal_extended)*K/len(t_of_simulation)
reconstructed_signal_spectrum    = np.fft.fft(reconstructed_signal)/len(t_of_simulation)


# Horizontal axis variables of the graphs
# fftfreq(window_length, sample spacing (inverse of sampling rate))
f_input     = np.fft.fftfreq(len(t_of_simulation), d=1/simulation_freq)

fig, axs = plt.subplots(5,2,figsize=(20, 25))
fig.subplots_adjust(left=0.5, bottom=0.5, 
                    right=None, top=None, 
                    wspace=0.3, hspace=0.7)

# First two graphs - Input signal and its spectrum
axs[0,0].plot(t_of_simulation, input_signal)
axs[0,0].set_title('Input Signal')
axs[0,0].set_xlabel('Time [s]')
axs[0,0].set_ylabel('Amplitude')
axs[0,0].grid(True)
col_1_vertical_axis_levels      = np.arange(-1,1.25,0.25)
col_1_horizontal_axis_levels    = np.arange(0,1.1,0.1)
axs[0,0].set_yticks(col_1_vertical_axis_levels)
axs[0,0].set_xticks(col_1_horizontal_axis_levels)
axs[0,0].set_xlim(0, time_interval_to_be_shown)

axs[0,1].stem(f_input, np.abs(input_signal_spectrum))
axs[0,1].set_title('Input Signal Spectrum')
axs[0,1].set_xlabel('Frequency [Hz]')
axs[0,1].set_ylabel('Magnitude')
axs[0,1].grid(True)
col_2_vertical_axis_levels      = np.arange(0,1.5,0.5)
col_2_horizontal_axis_levels    = np.arange(-100,100,20)
axs[0,1].set_yticks(col_2_vertical_axis_levels)
axs[0,1].set_xticks(col_2_horizontal_axis_levels)
axs[0,1].set_xlim(-frequency_interval_to_be_shown, frequency_interval_to_be_shown)

# Fllowing two graphs - impulse train and its spectrum
axs[1,0].stem(impulse_locations, impulse_train,linefmt='red')
axs[1,0].set_title('Impulse Train')
axs[1,0].set_xlabel('Time [s]')
axs[1,0].set_ylabel('Amplitude')
axs[1,0].grid(True)
axs[1,0].set_yticks([0,1])
axs[1,0].set_xticks(col_1_horizontal_axis_levels)
axs[1,0].set_xlim(0, time_interval_to_be_shown)

axs[1,1].stem(f_input, np.abs(impulse_train_spectrum))
axs[1,1].set_title('Impulse Train Spectrum')
axs[1,1].set_xlabel('Frequency [Hz]')
axs[1,1].set_ylabel('Magnitude')
axs[1,1].grid(True)
axs[1,1].set_yticks(col_2_vertical_axis_levels)
axs[1,1].set_xticks(col_2_horizontal_axis_levels)
axs[1,1].set_xlim(-frequency_interval_to_be_shown, frequency_interval_to_be_shown)

# Fllowing two graphs - impulse train with input signal and their spectrum
axs[2,0].plot(t_of_simulation, input_signal)
axs[2,0].stem(impulse_locations, sampled_signal,linefmt='red')
axs[2,0].set_title('Sampled Input Signal')
axs[2,0].set_xlabel('Time [s]')
axs[2,0].set_ylabel('Amplitude')
axs[2,0].grid(True)
axs[2,0].set_yticks(col_1_vertical_axis_levels)
axs[2,0].set_xticks(col_1_horizontal_axis_levels)
axs[2,0].set_xlim(0, time_interval_to_be_shown)

axs[2,1].stem(f_input, np.abs(sampled_input_signal_spectrum))
axs[2,1].set_title('Sampled Input Signal Spectrum')
axs[2,1].set_xlabel('Frequency [Hz]')
axs[2,1].set_ylabel('Magnitude')
axs[2,1].grid(True)
axs[2,1].set_yticks(col_2_vertical_axis_levels)
axs[2,1].set_xticks(col_2_horizontal_axis_levels)
axs[2,1].set_xlim(-frequency_interval_to_be_shown, frequency_interval_to_be_shown)



# Fllowing two graphs - STAIRCASE Reconstruction signal and its spectrum
axs[3,0].plot(t_of_simulation, reconstructed_signal)
axs[3,0].set_title('Staircase Reconstruction')
axs[3,0].set_xlabel('Time [s]')
axs[3,0].set_ylabel('Amplitude')
axs[3,0].grid(True)
axs[3,0].set_yticks(col_1_vertical_axis_levels)
axs[3,0].set_xticks(col_1_horizontal_axis_levels)
axs[3,0].set_xlim(0, time_interval_to_be_shown)

axs[3,1].stem(f_input, np.abs(reconstructed_signal_spectrum))
axs[3,1].set_title('Reconstructed Signal Spectrum')
axs[3,1].set_xlabel('Frequency [Hz]')
axs[3,1].set_ylabel('Magnitude')
axs[3,1].grid(True)
axs[3,1].set_yticks(col_2_vertical_axis_levels)
axs[3,1].set_xticks(col_2_horizontal_axis_levels)
axs[3,1].set_xlim(-frequency_interval_to_be_shown, frequency_interval_to_be_shown)


st.pyplot(fig)


'''
# Fllowing two graphs - signal through FIR - reconstructed signal and its spectrum
axs[3,0].plot(t_of_simulation, reconstructed_signal)
axs[3,0].set_title('Sampled signal through FIR')
axs[3,0].set_xlabel('Time [s]')
axs[3,0].set_ylabel('Amplitude')
axs[3,0].set_xlim(0, 1)
axs[3,0].grid(True)
axs[3,0].set_yticks(col_1_vertical_axis_levels)
axs[3,0].set_xticks(col_1_horizontal_axis_levels)

axs[3,1].stem(f_input, np.abs(reconstructed_signal_spectrum))
axs[3,1].set_title('Reconstructed Signal Spectrum')
axs[3,1].set_xlabel('Frequency [Hz]')
axs[3,1].set_ylabel('Magnitude')
axs[3,1].set_xlim(-100, 100)
axs[3,1].grid(True)
axs[3,1].set_yticks(col_2_vertical_axis_levels)
axs[3,1].set_xticks(col_2_horizontal_axis_levels)
'''


'''
sampled_signal_extended*np.sin(np.pi*t_of_sampling)/(np.pi*t_of_sampling)
sinc_function = np.sin(np.pi*t_of_sampling)/(np.pi*t_of_sampling)

reconstructed_signal = sinc_function*sampled_signal

reconstructed_signal = np.zeros_like(len(t_of_simulation))
for n in range(len(impulse_locations)):
    t_n = impulse_locations[n]
    value_of_the_sample = sampled_signal[n]
    new_sinc_signal = value_of_the_sample*np.sinc((t_of_simulation-t_n)/T_sampling)
    reconstructed_signal += new_sinc_signal
'''


'''
#METHOD in javascript code FIR
f_cutoff = sampling_freq/2
filter_coefficients  = signal.firwin(101, f_cutoff, fs=simulation_freq)
print('filter coefs are: ',filter_coefficients)
#filtfilt(numarator,denominator,data to be filtered)
reconstructed_signal = signal.filtfilt(filter_coefficients, 1,sampled_signal_extended)*K
print('Reconstructed signal is : ', reconstructed_signal)
'''


"""
imp = signal.unit_impulse(200, 'mid')
plt.plot(np.arange(0, 200), imp)



st.latex('''a \sin(2 \pi f t + phi)''')
fig,ax = subplots(figsize=(10,4))
xlim(0,0.010); ylim(-10, 10)
plot(t[0:100], signal[0:100])
xlabel('Time (seconds)')
st.pyplot(fig)
st.audio(signal,sample_rate=fe)

"""


'''
impulse_train_full = np.zeros_like(t_of_simulation)

total_samples = sampling_freq + 1
for n in range(total_samples):
    t_n = n*T_sampling
    if t_n < 1:
        index = int(np.round(t_n*simulation_freq))
        impulse_train_full[index] = 1   

only_the_ones_in_the_full_train = np.zeros_like(t_of_simulation)
for i in range(len(impulse_train_full)):
    only_the_ones_in_the_full_train.append(impulse_train_full[i] == 1)

print('filterin the ones', only_the_ones_in_the_full_train)
'''


'''
impulse_train_full   = np.ones_like(t_of_simulation)
all_values_are_sampled = impulse_train_full*input_signal

decomposition = int(np.round(simulation_freq/sampling_freq))
necessary_values_from_the_multiplication = all_values_are_sampled[::decomposition]
print(decomposition)
print(necessary_values_from_the_multiplication)
'''



'''
impulse_train_rearranged = impulse_train_full*input_signal
print('rearranged impulse train is', impulse_train_rearranged)
picking_period = int(np.round(simulation_freq/sampling_freq))
print('picking period is', picking_period)
impulse_train_reduced = impulse_train_rearranged[::picking_period]
print('reduced impulse train is', impulse_train_reduced)
print(impulse_train_rearranged[picking_period])
print(impulse_train_rearranged[2*picking_period+1])
print(impulse_train_rearranged[3*picking_period])
print(impulse_train_rearranged[4*picking_period])
'''




'''
#Should reduce the number of elements in the impulse train array
#We only want to plot the ones but not the zeros.
idx_impulse_present = impulse_train == 1
print(idx_impulse_present)
print(type(idx_impulse_present))
# this creates an array with the ones only
'''
