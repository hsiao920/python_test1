from numba import cuda, guvectorize, vectorize, void, int32, float64, uint32
import math
import numpy as np
np.random.seed(1)

@cuda.jit
def axpy(r, a, x, y):
    i = cuda.grid(1)
    if i < len(r):
        r[i] = a * x[i] + y[i]


def create_and_add_vectors(N):
    # Create input data and transfer to GPU
    x = np.random.random(N)
    y = np.random.random(N)
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_r = cuda.device_array_like(d_x)
    a = 4.5

    # Compute grid dimensions
    
    # An arbitrary reasonable choice of block size
    block_dim = 256
    # Enough blocks to cover the input
    grid_dim = math.ceil(len(d_x) / block_dim)

    # Launch the kernel
    axpy[grid_dim, block_dim](d_r, a, d_x, d_y)
    
    # Return the result
    return d_r.copy_to_host()



create_and_add_vectors(16)
