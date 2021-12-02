import numpy as np
import sys
sys.path.append("..")
import math


from optimization_algorithms.interface.nlp_solver import  NLPSolver


class SolverNGMethod(NLPSolver):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def evaluate(self, x):
        return self.problem.evaluate(x)[0][0]


    def solve(self):
        alpha = self.kwargs.get("alpha", 1)
        ro_is = self.kwargs.get("ro_is", 0.01)
        ro_plus = self.kwargs.get("ro_plus", 1.2)
        ro_minus = self.kwargs.get("ro_minus", 0.5)
        x = self.kwargs.get("x_init", self.problem.getInitializationSample())
        step_norm = math.inf
        count =0 
        while not np.all(step_norm<0.00009):
            y, J = self.problem.evaluate(x)
            fx = np.array(y)
            count =count +1
            H = self.problem.getFHessian(x)
            
            step =np.linalg.inv(H)@J.T@fx
            x_1 = x - step
            step_norm =  x_1 - x
            x = x_1
            print("iteration ", count, "x: ", x)

        y, J = self.problem.evaluate(x)
        return x



class SolverGradient(NLPSolver):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def evaluate(self, x):
        return self.problem.evaluate(x)[0][0]

    def solve(self):
        x = self.kwargs.get("x_init", self.problem.getInitializationSample())
        step_norm = math.inf
        count =0 
        while not np.all(step_norm<0.009):
            y, J = self.problem.evaluate(x)
            fx = np.array(y)

            step = J.T@fx
            
            x_1 = x - 0.01*step
            step_norm =  x_1 - x
            x = x_1
            count =  count+1
            print("iteration ", count, "x: ", x)

        y, J = self.problem.evaluate(x)
        return x








            
        