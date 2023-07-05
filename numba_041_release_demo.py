<<<<<<< HEAD
from numba import njit, config, __version__
from numba.extending import overload
import numpy as np
assert tuple(int(x) for x in __version__.split('.')[:2]) >= (0, 41)


if config.PYVERSION > (3, 4): # Only supported in Python >= 3.4
    
    @njit
    def strings_demo(str1, str2, str3):
        # strings, ---^  ---^   ---^
        # as arguments are now supported!
        
        # defining strings in compiled code also works
        def1 = 'numba is '
        
        # as do unicode strings
        def2 = '🐍⚡'
        
        # also string concatenation 
        print(str1 + str2)
        
        # comparison operations
        print(str1 == str2)
        print(str1 < str2)
        print(str1 <= str2)
        print(str1 > str2)
        print(str1 >= str2)
        
        # {starts,ends}with
        print(str1.startswith(str3))
        print(str2.endswith(str3))
        
        # len()
        print(len(str1), len(def2), len(str3))
        
        # str.find()
        print(str2.find(str3))
        
        # in
        print(str3 in str2)
        
        # slicing
        print(str2[1:], str1[:1])
        
        # and finally, strings can also be returned
        return '\nnum' + str1[1::-1] + def1[5:] + def2
    
    
    # run the demo
    print(strings_demo('abc', 'zba', 'a'))
    
=======
from numba import njit, config, __version__
from numba.extending import overload
import numpy as np
assert tuple(int(x) for x in __version__.split('.')[:2]) >= (0, 41)


if config.PYVERSION > (3, 4): # Only supported in Python >= 3.4
    
    @njit
    def strings_demo(str1, str2, str3):
        # strings, ---^  ---^   ---^
        # as arguments are now supported!
        
        # defining strings in compiled code also works
        def1 = 'numba is '
        
        # as do unicode strings
        def2 = '🐍⚡'
        
        # also string concatenation 
        print(str1 + str2)
        
        # comparison operations
        print(str1 == str2)
        print(str1 < str2)
        print(str1 <= str2)
        print(str1 > str2)
        print(str1 >= str2)
        
        # {starts,ends}with
        print(str1.startswith(str3))
        print(str2.endswith(str3))
        
        # len()
        print(len(str1), len(def2), len(str3))
        
        # str.find()
        print(str2.find(str3))
        
        # in
        print(str3 in str2)
        
        # slicing
        print(str2[1:], str1[:1])
        
        # and finally, strings can also be returned
        return '\nnum' + str1[1::-1] + def1[5:] + def2
    
    
    # run the demo
    print(strings_demo('abc', 'zba', 'a'))
    
>>>>>>> 5279899b69b29cd56fae64d120ae7e49e7589eaf
