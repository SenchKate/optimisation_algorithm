import numpy as np
import sys
import math 
sys.path.append("..")


from optimization_algorithms.interface.mathematical_program import  MathematicalProgram

class SQ(MathematicalProgram):
    def __init__(self, c,dim):
        self.c =c
        self.dim = dim
        self.c = np.diag([c**((i-1)/(dim-1)) for i in range(1,dim+1)])


    def evaluate(self, u):
        x = np.array(u)
        y = x.T @ self.c @ x
        grad = 2*self.c @ x
        return y,grad

    def getDimension(self) : 
        """
        """
        return self.dim 

    def getFHessian(self, x) : 
        """
        """
        return 2*self.c

    def getInitializationSample(self) : 
        """
        """
        return np.ones(2)

    def report(self , verbose ): 
        """
        """
        strOut = "It is SQ function"
        return  strOut

class Hole(MathematicalProgram):
     def __init__(self, dim, c, alpha):
        self.c =c
        self.dim = dim
        self.alpha = alpha
        self.C = np.diag([c**((i-1)/(dim-1)) for i in range(1,dim+1)])


     def evaluate(self, x):
        y = (x.T @ self.c @ x)/(self.alpha**2 + x.T @ self.c @ x)
        grad = (self.alpha**2)/((self.alpha **2 + x.T @ self.c @ x) **2)
        return y,grad

     def getDimension(self) : 
        """
        """
        return self.dim 

     def getFHessian(self, x) : 
        """
        """
        return -(self.alpha**2 * 2*(self.alpha **2 + x.T @ self.c @ x)*(2 *self.c @ x ))/(self.alpha **2 + x.T @ self.c @ x) **4

     def getInitializationSample(self) : 
        """
        """
        return np.ones(2)

     def report(self , verbose ): 
        """
        """
        strOut = "It is HOLE function"
        return  strOut



class Phi(MathematicalProgram):
     def __init__(self, a, c):
        self.a =a
        self.c = c


     def evaluate(self, x):
        phi = np.array([math.sin(self.a * x[0]), math.sin(self.c* self.a * x[1]), 2*x[0], 2*self.c * x[1]] )
        self.J = 2 * np.array([[self.a * math.cos(self.a * x[0]) , 0], [0, self.c* self.a * math.cos(self.c* self.a * x[1])],  [2,0],[0, 2*self.c ]])
        return phi ,self.J

     def getDimension(self) : 
        """
        """
        return self.dim 

     def getFHessian(self, x) : 
        """
        """
        return 2*self.J.T @ self.J

     def getInitializationSample(self) : 
        """
        """
        return np.ones(2)

     def report(self , verbose ): 
        """
        """
        strOut = "It is Phi function"
        return  strOut
