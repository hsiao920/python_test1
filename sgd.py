<<<<<<< HEAD
import numpy as np

def gradient_descent( gradient,start,learn_rate,n_iter=50,tolerance=1e-06):
    vector=start
    for _ in range(n_iter):
        diff=-learn_rate*gradient(vector)
        if np.all(np.abs(diff)<=tolerance):
            break
        vector+=diff
    return vector
=======
import numpy as np

def gradient_descent( gradient,start,learn_rate,n_iter=50,tolerance=1e-06):
    vector=start
    for _ in range(n_iter):
        diff=-learn_rate*gradient(vector)
        if np.all(np.abs(diff)<=tolerance):
            break
        vector+=diff
    return vector
>>>>>>> 5279899b69b29cd56fae64d120ae7e49e7589eaf
