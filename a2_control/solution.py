import sys
import math
from telnetlib import TTYLOC
import numpy as np

sys.path.append("..")
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT


class LQR(MathematicalProgram):
    """
    Parameters
    K integer
    A in R^{n x n}
    B in R^{n x n}
    Q in R^{n x n} symmetric
    R in R^{n x n} symmetric
    yf in R^n

    Variables
    y[k] in R^n for k=1,...,K
    u[k] in R^n for k=0,...,K-1

    Optimization Problem:
    LQR with terminal state constraint

    min 1/2 * sum_{k=1}^{K}   y[k].T Q y[k] + 1/2 * sum_{k=1}^{K-1}      u [k].T R u [k]
    s.t.
    y[1] - Bu[0]  = 0
    y[k+1] - Ay[k] - Bu[k] = 0  ; k = 1,...,K-1
    y[K] - yf = 0

    Hint: Use the optimization variable:
    x = [ u[0], y[1], u[1],y[2] , ... , u[K-1], y[K] ]

    Use the following features:
    1 - a single feature of types OT.f
    2 - the features of types OT.eq that you need
    """

    def __init__(self, K, A, B, Q, R, yf):
        self.K = K
        self.A =A
        self.B =B
        self.Q = Q
        self.R = R
        self.yf=yf
        self.dim = R.shape[0]
        """
        Arguments
        -----
        T: integer
        A: np.array 2-D
        B: np.array 2-D
        Q: np.array 2-D
        R: np.array 2-D
        yf: np.array 1-D
        """
        # in case you want to initialize some class members or so...

    def evaluate(self, x):
        """
        See also:
        ----
        MathematicalProgram.evaluate
        """
        x = x.reshape((-1,self.dim))
        index_u = list(range(0,x.shape[0],2))
        index_y = list(range(1,x.shape[0],2))
        u = x[index_u,:]
        u = u.T
       
        y = x[index_y,:]
        x = x.reshape((1,-1))
        y=y.T
        phi=0
        for i in range(y.shape[1]):
            phi += 0.5*(u[:,i].T @ self.R @ u[:,i])+ 0.5*(y[:,i].T @ self.Q @ y[:,i])
       
        y = y.T
        u =u.T
        constr = np.array([])
        constr = np.append(constr, y[0] - self.B@ u[0])
        s_c = np.array([])
        for i in range(0,y.shape[0]-1):
            s_c = np.append(s_c, (y[i+1]-self.A @ y[i] -self.B@u[i+1] ))
        
        constr = np.append(constr, s_c)
        constr = np.append(constr,y[y.shape[0]-1] - self.yf ) 
        
        phi =  np.array([phi])
        phi = np.append(phi, constr)
        J = np.zeros(((self.K +1)*self.dim+1, x.size))
        z = np.zeros(( int(x.size/2),self.dim))
        for i in range(int(x.size/4)):
            z[i] = (u[i].T @ self.R)
            z[i+1]= y[i].T @ self.Q
        z = z.flatten()
        J[0] = z
       
        for i in range(0,self.K-1):
            for z in range(self.dim):
                J[i+1+z,2*i*self.dim+z]-=self.B[z].sum()
                J[i+1+z,(2*(i+1)+1)*self.dim+z] += 1
                J[i+1+z,(2*i+1)*self.dim+z]-=self.A[z].sum()

        for z in range(self.dim):
            J[(self.K )*self.dim+1+z,x.size - self.dim+z]=1 
        
        return phi, J


        
        
       

    def getFHessian(self, x):
        H = np.zeros((self.K*2*self.dim,self.K*2*self.dim))
        for i in range(int(self.K)):
            for z in range (self.dim):
                H[2*i*self.dim+z,2*i*self.dim+z] = self.R[z].sum()
                H[(2*i+ 1)*self.dim +z,(2*i+ 1)*self.dim +z] = self.Q[z].sum()
                
        if (2*i+ 1)*self.dim +z < self.K*2*self.dim-1:
             for z in range (self.dim):
                H[self.K*2*self.dim - self.dim+z,self.K*2*self.dim - self.dim+z] = self.R[z].sum()
        return H

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        return self.K *2*self.dim

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return np.zeros(self.getDimension())

    def getFeatureTypes(self):
        """
        returns
        -----
        output: list of feature Types
        See Also
        ------
        MathematicalProgram.getFeatureTypes
        """
        return  [OT.f]+ [OT.eq]*(self.K +1)*self.dim


"""
        J = np.zeros(((self.K +1)*self.dim+1, x.size))
        z = np.zeros(( int(x.size/2),self.dim))
        for i in range(int(x.size/4)):
            z[i] = (u[i].T @ self.R)
            z[i+1]= y[i].T @ self.Q
        z = z.flatten()
        J[0] = z
        z=self.B@ np.ones(*u[0].shape)
        J[1,1*self.dim]=1
        J[1,0] = -z[0]
        
        for i in range(0,y.shape[0]-1):
            c=self.A@ np.ones(*y[i].shape)
            a=self.B@ np.ones(*u[i+1].shape)
            J[i+2,((2*i+1)+1)*self.dim] = 1
            J[i+2,(2*i+1)*self.dim] = c[0]
            J[i+2,(2*i)*self.dim] = a[0]
        J[(self.K +1)*self.dim,x.size-self.dim-1]=1

        for i in range(y.shape[1]):
            J[0,0] += (u[i].T@ self.R)
        for i in range(y.shape[1]):
            J[1,0] += (y[i].T@ self.Q)
        J[0,1] = self.B@ np.ones(*(u[0]).shape)
        J[1,1] = np.ones(*(y[0]).shape)
        for i in range(0,y.shape[0]-1):
            J[0,i+3]=  -self.B@np.ones(*(u[i]).shape)
            J[0,i+2]=  -self.A@np.ones(*(y[i]).shape)
        J[1,(self.K +1)*self.dim]= np.ones(self.dim,)
    """