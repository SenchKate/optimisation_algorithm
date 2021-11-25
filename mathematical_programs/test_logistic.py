import numpy as np
import unittest
import sys
import math
sys.path.append("..")

# import the test classes



# from optimization_algorithms.utils.finite_diff import *
# from optimization_algorithms.interface.mathematical_program import MathematicalProgram


from utils.finite_diff import *
from interface.mathematical_program import MathematicalProgram


from logistic import Logistic

class testLogistic(unittest.TestCase):
    """
    test on problem A
    """
    problem = Logistic

    def testConstructor(self):
        p = self.problem()

    def testValue1(self):
        problem = self.problem()
        # in this configuration, p = pr
        # todo: test the cost
        phi, J = problem.evaluate(problem.xopt)
        self.assertTrue( np.allclose(phi, np.zeros(problem.num_points)))

    # def testValue2(self):
    #     problem = self.generateProblem()
    #     # in this configuration, q = q0
    #     # todo: test the cost
    #     x =  np.zeros(3)
    #     phi, J = problem.evaluate(x)
    #     self.assertTrue( np.allclose(phi[:2], np.array( [1.5 + 1./3. -.5 , -2./3.] )))
    #     self.assertTrue( np.allclose( phi[2:] , np.zeros(3)))

    def testJacobian(self):
        problem = self.problem()
        x =  np.array([-1,.5,1])  
        flag , _ , _= check_mathematical_program(problem.evaluate,x, 1e-5, True)
        self.assertTrue(flag)


# usage:
# print results in terminal
# python3 test.py
# store results in file 
# python3 test.py out.log

if __name__ == "__main__":
    if len(sys.argv) == 2 :
        log_file = sys.argv.pop()
        with open(log_file, "w") as f:
           runner = unittest.TextTestRunner(f, verbosity=2)
           unittest.main(testRunner=runner)
    else:
        unittest.main()



