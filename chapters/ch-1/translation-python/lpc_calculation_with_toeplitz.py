
import numpy as np
from scipy.linalg import solve_toeplitz

def lpc_toeplitz(frame, order):
    # Autocorrelation vector r=[R[0] R[1] R[2] ... R[p]]
    r = [sum(frame[n] * frame[n+p] for n in range(len(frame)-p)) for p in range(order + 1)]
    # To specify the Toeplitz matrix, only the first column and the first row are needed.
    # Since autocorrelation matrix is symmetric the first row and first colums are the same.
    # first_column = first_row = r[:-1], right_hand_side = r[1:]
    a_rest = solve_toeplitz((r[:-1],r[:-1]), -np.array(r[1:]))
    # Inserting 1 as the 0th element
    a = np.insert(a_rest, 0, 1)
    sigma_squared = np.dot(a, r)/len(frame)
    return a, sigma_squared
