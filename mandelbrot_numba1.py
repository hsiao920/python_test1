<<<<<<< HEAD
import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer
#from numba import jit
from numba import njit, generated_jit, types


@generated_jit
def select(x):
    if isinstance(x, types.Float):
        def impl(x):
            return x + 1
        return impl
    elif isinstance(x, types.UnicodeType):
        def impl(x):
            return x + " the number one"
        return impl
    else:
        raise TypeError("Unsupported Type")


@njit
def mandel(x, y, max_iters):
  """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
  """
  c = complex(x, y)
  z = 0.0j
  for i in range(max_iters):
    z = z*z + c
    if (z.real*z.real + z.imag*z.imag) >= 4:
      return i

  return max_iters

@njit
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
  height = image.shape[0]
  width = image.shape[1]

  pixel_size_x = (max_x - min_x) / width
  pixel_size_y = (max_y - min_y) / height
    
  for x in range(width):
    real = min_x + x * pixel_size_x
    for y in range(height):
      imag = min_y + y * pixel_size_y
      color = mandel(real, imag, iters)
      image[y, x] = color

      

image = np.zeros((1024, 1536), dtype = np.uint8)
start = timer()
create_fractal(-2.0, 1.0, -1.0, 1.0, image, 20) 
dt = timer() - start
print ("Mandelbrot created in %f s" % dt)
imshow(image)
show()

start = timer()
create_fractal(-2.0, -1.7, -0.1, 0.1, image, 20)
dt = timer() - start
print ("Mandelbrot created in %f s" % dt)
imshow(image)
show()

=======
import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer
#from numba import jit
from numba import njit, generated_jit, types


@generated_jit
def select(x):
    if isinstance(x, types.Float):
        def impl(x):
            return x + 1
        return impl
    elif isinstance(x, types.UnicodeType):
        def impl(x):
            return x + " the number one"
        return impl
    else:
        raise TypeError("Unsupported Type")


@njit
def mandel(x, y, max_iters):
  """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
  """
  c = complex(x, y)
  z = 0.0j
  for i in range(max_iters):
    z = z*z + c
    if (z.real*z.real + z.imag*z.imag) >= 4:
      return i

  return max_iters

@njit
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
  height = image.shape[0]
  width = image.shape[1]

  pixel_size_x = (max_x - min_x) / width
  pixel_size_y = (max_y - min_y) / height
    
  for x in range(width):
    real = min_x + x * pixel_size_x
    for y in range(height):
      imag = min_y + y * pixel_size_y
      color = mandel(real, imag, iters)
      image[y, x] = color

      

image = np.zeros((1024, 1536), dtype = np.uint8)
start = timer()
create_fractal(-2.0, 1.0, -1.0, 1.0, image, 20) 
dt = timer() - start
print ("Mandelbrot created in %f s" % dt)
imshow(image)
show()

start = timer()
create_fractal(-2.0, -1.7, -0.1, 0.1, image, 20)
dt = timer() - start
print ("Mandelbrot created in %f s" % dt)
imshow(image)
show()

>>>>>>> 5279899b69b29cd56fae64d120ae7e49e7589eaf
