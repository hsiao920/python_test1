<<<<<<< HEAD
import unittest

addtwo=lambda x:x+2

class LambdaTest(unittest.TestCase):
    def test_add_two(self):
        self.assertEqual(addtwo(2),4)

    def test_add_two_point_two(self):
        self.assertEqual(addtwo(2.2),4.2)

    def test_add_three(self):
        #should fail
        self.assertEqual(addtwo(4),6)

if __name__=='__main__':
    unittest.main(verbosity=2)
    
print(__name__)
=======
import unittest

addtwo=lambda x:x+2

class LambdaTest(unittest.TestCase):
    def test_add_two(self):
        self.assertEqual(addtwo(2),4)

    def test_add_two_point_two(self):
        self.assertEqual(addtwo(2.2),4.2)

    def test_add_three(self):
        #should fail
        self.assertEqual(addtwo(4),6)

if __name__=='__main__':
    unittest.main(verbosity=2)
    
print(__name__)
>>>>>>> 5279899b69b29cd56fae64d120ae7e49e7589eaf
