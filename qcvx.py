import numpy as np
import cvxpy as cp

def partial_trace(A, dims, trace_list, discard=True):
    '''partial trace and 
    possibly replace the traced-out systems 
    with maximally mixed state
    '''
    no_trace_list = [i for i in range(len(dims)) if i not in trace_list]
    new_order = no_trace_list + trace_list
    dims = np.array(dims)
    d_trace = np.prod(dims[trace_list])
    d_no_trace = np.prod(dims[no_trace_list])
    
    # calculate the permutation vector
    idx = np.array(range(np.prod(dims)))
    p = np.einsum(idx.reshape(dims), range(len(dims)), new_order).flatten()
    # use indexing to permute the subsystems
    A = cp.partial_trace(A[p][:, p], [d_no_trace, d_trace], 1)
    if discard:
        return A
    else:
        # the inversion of the permutation
        p_inv = np.einsum(idx.reshape(dims[new_order]), new_order).flatten()
        return cp.kron(A, np.eye(d_trace) / d_trace)[p_inv][:, p_inv]