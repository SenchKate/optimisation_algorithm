import sys
import math
import numpy as np

sys.path.append("..")
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT


class RobotTool(MathematicalProgram):
    """
    """

    def __init__(self, q0, pr, l):
        """
        Arguments
        ----
        q0: 1-D np.array
        pr: 1-D np.array
        l: float
        """
        self.q0 = q0
        self.pr= pr
        self.l = l
        # in case you want to initialize some class members or so...

    def evaluate(self, x):
        """
        See also:
        ----
        MathematicalProgram.evaluate
        """
        pb = np.zeros((2,))
        pb[0] = math.cos(x[0]) + 1/2*math.cos(x[0]+x[1])+1/3*math.cos(x[0]+x[1]+x[2])
        pb[1] = math.sin(x[0]) + 1/2*math.sin(x[0]+x[1])+1/3*math.sin(x[0]+x[1]+x[2])
        y =np.sqrt((pb-self.pr).T@(pb-self.pr)+self.l*((x- self.q0).T@(x- self.q0)))
        # add the main code here! E.g. define methods to compute value y and Jacobian J
        # y = ...
        J = np.zeros((2,3))

        J[0,0] = (pb[0]-self.pr[0])*2*(-math.sin(x[0])-1/2*math.sin(x[0]+x[1])-math.sin(x[0]+x[1]+x[2]))+self.l*2*(x[0]-self.q0[0])
        J[0,1] = (pb[0]-self.pr[0])*2*(-1/2*math.sin(x[0]+x[1])-math.sin(x[0]+x[1]+x[2]))+self.l*2*(x[1]-self.q0[1])
        J[0,2] = (pb[0]-self.pr[0])*2*(-math.sin(x[0]+x[1]+x[2]))+self.l*2*(x[2]-self.q0[2])
        J[1,0] = (pb[1]-self.pr[1])*2*(math.cos(x[0]) + 1/2*math.cos(x[0]+x[1]) + 1/2*math.cos(x[0]+x[1]+x[2]))+self.l*2*(x[0]-self.q0[0])
        J[1,1] = (pb[1]-self.pr[1])*2*(1/2*math.cos(x[0]+x[1]) + 1/2*math.cos(x[0]+x[1]+x[2]))+self.l*2*(x[1]-self.q0[1])
        J[1,2] = (pb[1]-self.pr[1])*2*(1/2*math.cos(x[0]+x[1]+x[2]))+self.l*2*(x[2]-self.q0[2])
        # y is a 1-D np.array of dimension m
        # J is a 2-D np.array of dimensions (m,n)
        # where m is the number of features and n is dimension of x
        return  np.array([y]).reshape((1,1))  , J

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        # return the input dimensionality of the problem (size of x)
        # return

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        # return ...

    def getFeatureTypes(self):
        """
        returns
        -----
        output: list of feature Types
        See Also
        ------
        MathematicalProgram.getFeatureTypes
        """
        return [OT.sos] * 5
