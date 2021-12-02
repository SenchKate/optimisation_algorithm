import math
import sys
import numpy as np

sys.path.append("..")
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT


class AntennaPlacement(MathematicalProgram):
    """
    """

    def __init__(self, P, w):
        self.p =np.array(P)
        self.w =w 
        """
        Arguments
        ----
        P: list of 1-D np.arrays
        w: 1-D np.array
        """
        # in case you want to initialize some class members or so...
    



    def returnForX(self,x, func):
        first= func(0,x)
        second = func(1,x)
        return first+second
        
    def Jacobian(self,x):
        J=np.zeros((2,2))
        J[0,1] =J[1,0]= 0
        J[0,0]=self.func(0,x[0])*(-((self.squareSum(0,x[0])).sum()))*2 + self.func(1,x[0])*(-(self.squareSum(1,x[0]).sum()))*2
        J[1,1]=self.func(0,x[1])*(-(self.squareSum(0,x[1]).sum()))*2 + self.func(1,x[1])*(-(self.squareSum(1,x[1]).sum()))*2
        return J

    def evaluate(self, x):
        """
        See also:
        ----
        MathematicalProgram.evaluate
        """
        def func(forI,x):
            return -self.w[forI] * np.exp((-(x-self.p[forI])**2).sum())
        def Jacobian(forW,x,forX):
            return func(forW,x)*(-(x[forX]-self.p[forW,forX]))*2
        J = np.zeros((2,))
        J[0] = Jacobian(0,x,0)+ Jacobian(1,x,0)
        J[1] = Jacobian(1,x,1)+ Jacobian(0,x,1)
        y=np.zeros([1])
        y[0] = self.returnForX(x,func)
        return y , np.array([J])

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        # return the input dimensionality of the problem (size of x)
        # return ...
        return 2

    def getFHessian(self, x):
        """
        See Also
        ------
        MathematicalProgram.getFHessian
        """
        def func(forI,x):
            return -self.w[forI] * np.exp((-(x-self.p[forI])**2).sum())
        H=np.zeros((2,2))
        H[0,0] = 4*(func(0,x))*((x[0]-self.p[0,0])**2) - 2*func(0,x) - 2*func(1,x) + 4*(func(1,x))*((x[0]-self.p[1,0])**2)
        H[1,0] =H[0,1]= func(0,x)*4*(x[0]-self.p[0,0])*(x[1]-self.p[0,1]) + func(1,x)*4*(x[0]-self.p[1,0])*(x[1]-self.p[1,1])
        H[1,1] = 4*(func(0,x))*(x[1]-self.p[0,1])**2 - 2*func(0,x) - 2*func(1,x) +4*(func(1,x))*(x[1]-self.p[1,1])**2
        return H

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        x0 = (self.p).sum(axis = 1)/(self.p.shape[1])
        return x0

    def getFeatureTypes(self):
        """
        returns
        -----
        output: list of feature Types

        """
        return [OT.f]
