import numpy as np
from numba import prange # import parallel range
from numba import cuda
from numba import *


# decorate a function with `parallel=True` as usual
@njit(parallel=True)
def test(x):
    n = x.shape[0]
    a = np.sin(x)                      # parallel array expression
    b = np.cos(a * a)                  # parallel array expression
    acc = 0                            
    for i in prange(n - 2):            # user defined parallel loop
        for j in prange(n - 1):        # user defined parallel loop
            acc += b[i] + b[j + 1]     # parallel reduction
    return acc

# run the function
test(np.arange(10))

# access the diagnostic output via the new `parallel_diagnostics` method on the dispatcher
test.parallel_diagnostics(level=4)
