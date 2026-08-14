
import numpy as np
import matplotlib.pyplot as plt

# a_coefficients = [1 a1 a2 ... ap] form A(z) and 1/A(z) as follows:
# A(z) = 1 + a1*z^-1 + a2*z^-2 + ... + ap*z^-p
# 1/A(z) = 1/(1 + a1*z^-1 + a2*z^-2 + ... + ap*z^-p)

def plot_zplane(b, a):
    """
    Replicates MATLAB's zplane(b, a) functionality.
    b: numerator coefficients
    a: denominator coefficients
    """
    # first equate the lengths by padding zeros, we shouldn'T miss the zeros.
    max_len = max(len(b), len(a))
    b_padded = np.pad(b, (0, max_len - len(b)), 'constant')
    a_padded = np.pad(a, (0, max_len - len(a)), 'constant')
    # Then find the roots:
    # Notice there will be zeros on the origin because b=[1, 0, ..., 0]
    zeros = np.roots(b_padded)
    poles = np.roots(a_padded)
    
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)
    unit_circle = plt.Circle((0,0), 1, color='blue',fill=False, linestyle='--')
    ax.add_patch(unit_circle)
    plt.axvline(0, color='blue',linestyle='--')
    plt.axhline(0, color='blue',linestyle='--')

    plt.plot(np.real(zeros), np.imag(zeros), 'o', markersize=8, markerfacecolor='None',
            color='blue', label='Zeros')
    
    plt.plot(np.real(poles), np.imag(poles), 'x', markersize=8, 
            color='blue', label='Poles')
        
    plt.title('Pole-Zero Plot (Z-Plane)')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.axis('equal') 
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.grid(True)
    plt.legend()
