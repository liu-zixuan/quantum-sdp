import numpy as np

# Pauli matrices
sigma_0 = np.array([[1., 0.], [0., 1.]], dtype=complex)
sigma_x = np.array([[0., 1.], [1., 0.]], dtype=complex)
sigma_y = np.array([[0., -1.j], [1.j, 0.]], dtype=complex)
sigma_z = np.array([[1., 0.], [0., -1.]], dtype=complex)

def basis(d, i):
    '''The standard basis
    '''
    v = np.zeros(d)
    v[i] = 1
    return v

def mbasis(d, i):
    '''The standard basis of matrices
    '''
    M = np.zeros((d,d))
    M[i,i] = 1
    return M