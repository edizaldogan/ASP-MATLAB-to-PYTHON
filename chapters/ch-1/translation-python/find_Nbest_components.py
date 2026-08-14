import numpy as np


def find_Nbest_components(sig, codebook_vectors, N):
    """
    Finds the N best codebook components to represent the target signal.
    The residual error is minimized as:
    error = signal - codebook_vectors[:, indices] * gains
    """
    # M: number of rows (frame_shift = 40), L: number of columns (codebook_size = 512)
    M, L = codebook_vectors.shape
    
    # Calculate the norm of each vector once to optimize the loop
    codebook_norms = np.linalg.norm(codebook_vectors, axis=0)
    
    gains = np.zeros(N)
    # Forcing the index array to 'int' to prevent type conversion errors
    indices = np.zeros(N, dtype=int) 
    
    # Copy the signal since we will mutate it inside the greedy loop
    current_signal = sig.copy()
    
    for k in range(N):
        max_norm = 0
        best_j = 0
        
        for j in range(L):
            # MATLAB: beta = codebook_vectors(:,j)' * signal
            beta = np.dot(codebook_vectors[:, j], current_signal)
            
            if codebook_norms[j] != 0:
                component_norm = np.abs(beta) / codebook_norms[j]
            else:
                component_norm = 0
                
            if component_norm > max_norm:
                gains[k] = beta / (codebook_norms[j]**2)
                best_j = j
                max_norm = component_norm
                
        indices[k] = best_j
        
        # Greedy Algorithm: Subtract the chosen component's effect from the signal
        current_signal = current_signal - gains[k] * codebook_vectors[:, best_j]
        
    return gains, indices