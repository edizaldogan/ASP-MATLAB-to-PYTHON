
def autocorrelation_calculation(input_frame, order):
    '''
    Takes in the 240 element frame and calculates the autocorrelation
    values R[0], R[1],..., R[p] where p is the order.
    '''
    autocorrelation_vector = []
    for p in range(order + 1):
        sum = 0
        for n in range(len(input_frame)-p):
            sum = sum + input_frame[n]*input_frame[n+p]
        autocorrelation_vector.append(float(sum))
    return autocorrelation_vector

def lpc_calculation(input_frame_for_letter_e, order):
    '''
    For a given frame we try to find the optimal ai coefficients that 
    minimizes the expectation of the residual energy argmin(E[e^2[n]]).
    This function solves the Yule-Walker equations by brute-force.
    '''
    autocorrelation_vector = autocorrelation_calculation(input_frame_for_letter_e, order)

    R = []
    for i in range(1,order+1):
        R_row_i = []
        for j in range(1,order+1):
            R_row_i.append(autocorrelation_vector[abs(i-j)])
        R.append(R_row_i)
    
    a_v_o = autocorrelation_vector.copy()
    for i in range(len(autocorrelation_vector)):
        autocorrelation_vector[i] = -autocorrelation_vector[i]
    autocorrelation_vector.pop(0)
    a_v_rhs = autocorrelation_vector

    # Gaussian Elimination Algorithm
    for i in range(order):
        coef_of_the_a_to_be_eliminated = R[i][i]
        for j in range(i+1,order):
            multiplier = R[j][i]/coef_of_the_a_to_be_eliminated
            for k in range(i, order):
                R[j][k] = R[j][k] - (multiplier*R[i][k])
            a_v_rhs[j] = a_v_rhs[j] - (multiplier*a_v_rhs[i])

    a_coefficients = [0]*order
    
    for i in range(order-1, -1, -1):
        current_sum = 0
        for j in range(i + 1, order):
            current_sum = current_sum + (R[i][j]*a_coefficients[j])
        a_coefficients[i] = (a_v_rhs[i] - current_sum)/R[i][i]
        
    a_coefficients.insert(0, 1)
    sigma_squared = 0
    for i in range(order+1):
        sigma_squared = sigma_squared + (a_coefficients[i]*a_v_o[i])
    sigma_squared = sigma_squared/len(input_frame_for_letter_e)

    return a_coefficients, sigma_squared
