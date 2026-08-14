
DEBUG_MODE = False

def autocorrelation_calculation(input_frame, order):
    '''
    Takes in the 240 element frame and calculates the autocorrelation
    values R[0], R[1],..., R[p] where p is the order.
    '''
    # Autocorrelation vector will be in the form: r=[R[0] R[1] R[2] ... R[p]]
    autocorrelation_vector = []
    # p (the order) goes from 0 to p.
    for p in range(order + 1):
        sum = 0
        # n goes from 0 to 240-p.
        for n in range(len(input_frame)-p):
            # By definition, we multiply the signal itself by its 
            # k-unit shifted version and add the results.
            sum = sum + input_frame[n]*input_frame[n+p]
        autocorrelation_vector.append(float(sum))
    return autocorrelation_vector

def lpc_calculation(input_frame_for_letter_e, order):
    '''
    For a given frame we try to find the optimal ai coefficients that 
    minimizes the expectation of the residual energy argmin(E[e^2[n]]).
    This function solves the Yule-Walker equations by brute-force.
    Yule-Walker equations are in the following matrix format: R*a=r.
    '''
    
    # Create the autocorrelation vector containing R[0] to R[p]
    autocorrelation_vector = autocorrelation_calculation(input_frame_for_letter_e, order)

    # CHECK POINT - autocorrelation vector
    if DEBUG_MODE:
        counter = 0
        print('autocorrelation vector is: ')
        for element in autocorrelation_vector:
            print('R[',counter,'] = ', element, sep='')
            counter += 1
        print('')
    
    R = [] # pxp matrix
    # start from row 1 to row p
    # Remember: Autocorrelation vector = [R[0] R[1] R[2] ... R[p]]
    for i in range(1,order+1):
        R_row_i = []
        for j in range(1,order+1):
            # Row i=1 of R is R[0], R[-1](or R[1]), ..., R[1-p] (or R[p-1])
            # Row i=2 of R is R[1], R[0], R[-1](or R[1]), ..., R[2-p] (or R[p-2])
            # ...
            # Row i=p of R is R[p-1], R[p-2], R[p-3], ..., R[0]
            # It is enough to check the absolute value of the difference i and j
            # because R[k]=R[-k] is satisfied for any k.
            R_row_i.append(autocorrelation_vector[abs(i-j)])
        R.append(R_row_i)
    
    # CHECK POINT - autocorrelation matrix R
    if DEBUG_MODE:
        row_counter = 0
        print('autocorrelation matrix (R) is: ')
        for row in R:
            col_counter = 0
            for element in row:
                value = abs(row_counter-col_counter)
                print(f'R[{row_counter}][{col_counter}]=R[{value}]={element:>6.3f}', end=' | ')
                col_counter += 1
            print('')
            row_counter += 1
        print('')

    # Yule-Walker Equations expanded form: 
    # a1*R[k-1] + a2*R[k-2] +...+ ap*R[k-p] = -R[k] where k=1,2,...,order
    # In matrix form, right hand side is: -[R[1] R[2] ... R[p]]

    # 11 elements original vector, we will use it later
    a_v_o = autocorrelation_vector.copy()

    # To obtain the right hand side, we need to negate the autocorrelation vector:
    for i in range(len(autocorrelation_vector)):
        autocorrelation_vector[i] = -autocorrelation_vector[i]
    
    # CHECK POINT - Negated autocorrelation vector
    if DEBUG_MODE:
        counter = 0
        print('Negated autocorrelation vector is: ')
        for element in autocorrelation_vector:
            print('R[',counter,'] = ', element, sep='')
            counter += 1
        print('')

    # Exclude the first term R[0]
    autocorrelation_vector.pop(0) # R[0] is taken out
    # Call this new autocorrelation_vector as a_v_rhs
    a_v_rhs = autocorrelation_vector # px1 vector
    # a_v_rhs = -[R[1] R[2] ... R[p]]

    # CHECK POINT - Right hand side vector
    if DEBUG_MODE:
        counter = 1
        print('Right hand side vector (a_v_rhs) is: ')
        for element in a_v_rhs:
            print('R[',counter,'] = ', element, sep='')
            counter += 1
        print('')

    '''
    Now that R(autocorrelation matrix) and a_v_rhs(autocorrelation vector) are 
    formed, we start eliminating the unknowns by replacing them in terms of the 
    rest of the unknowns using Gaussian elimination.   
    '''

    # Gaussian Elimination Algorithm
    for i in range(order): # picking the i_th element of i_th row
        coef_of_the_a_to_be_eliminated = R[i][i]
        for j in range(i+1,order): # finding multipliers under row i
            multiplier = R[j][i] / coef_of_the_a_to_be_eliminated
            for k in range(i, order): # executing subtraction of rows
                R[j][k] = R[j][k] - (multiplier * R[i][k])
            a_v_rhs[j] = a_v_rhs[j] - (multiplier * a_v_rhs[i])

    # Obtaining a_coefficients = [a1 a2 ... ap]:
    a_coefficients = [0]*order # px1 lpc coefficients vector
    
    # Begin from the right
    for i in range(order-1, -1, -1):
        current_sum = 0
        # Notice that below will be skipped in the first loop
        for j in range(i + 1, order): # only right side concerns us.
            current_sum = current_sum + (R[i][j] * a_coefficients[j])
            
        # Rearrange the Yule-Walker Equation, leave lpc coef. alone.
        a_coefficients[i] = (a_v_rhs[i] - current_sum) / R[i][i]
        
    # By definition of LPC, the first coefficient a0 is always 1.
    # We insert 1 at the very beginning of our array.
    a_coefficients.insert(0, 1) # (p+1)x1 vector
    # Obtaining a_coefficients = [a0 a1 a2 ... ap]:

    # Residual energy: E = a0*R[0] + a1*R[1] + a2*R[2] + ... + ap*R[p]
    sigma_squared = 0
    for i in range(order+1):
        sigma_squared = sigma_squared + (a_coefficients[i] * a_v_o[i])
    # MATLAB always divide the summation by the length of the window:
    sigma_squared = sigma_squared/len(input_frame_for_letter_e)

    return a_coefficients, sigma_squared
