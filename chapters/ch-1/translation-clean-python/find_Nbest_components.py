
import numpy as np

def find_Nbest_components(sig, codebook_vectors, N):
    """
    This function finds the N best components of signal from the 
    vectors in codebook_vectors, so that the residual error:
    error=signal- codebook_vectors[indices]*gains 
    is minimized.
    Components are found one-by-one using a greedy algorithm. 
    When components in codebook_vectors are not orthogonal, 
    the search is therefore suboptimal. 
    """
 
    M, L = codebook_vectors.shape
    
    codebook_norms = np.linalg.norm(codebook_vectors, axis=0)
    
    gains = np.zeros(N)
    indices = np.zeros(N, dtype=int) 
    
    current_signal = sig.copy()
    
    for k in range(N):
        max_norm = 0
        best_j = 0
        for j in range(L):
            beta = np.dot(codebook_vectors[:, j], current_signal) 
            if codebook_norms[j] != 0:
                component_norm = np.abs(beta)/codebook_norms[j]
            else:
                component_norm = 0

            if component_norm > max_norm:
                gains[k] = beta / (codebook_norms[j]**2)
                best_j = j
                max_norm = component_norm
                
        indices[k] = best_j
        
        current_signal = current_signal - gains[k]*codebook_vectors[:, best_j]
        
    return gains, indices
