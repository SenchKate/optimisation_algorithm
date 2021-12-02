import numpy as np
import sys
sys.path.append("..")


from optimization_algorithms.interface.mathematical_program_traced import  MathematicalProgramTraced
from a0_quadratic_function.problems import SQ, Hole, Phi
from a0_quadratic_function.solution import SolverNGMethod, SolverGradient
from optimization_algorithms.mathematical_programs.quadratic_identity_2 import QuadraticIdentity2


class Test:

     def __init__(self, problem, solver):
        self.problem = problem
        self.solver = solver

     def test_convergence(self, plot=True):
        """
        check that student solver converges
        """
        assert self.problem and self.solver
        
        self.solver.setProblem(self.problem)
        output =  self.solver.solve()
        
        last_trace = self.problem.trace_x[-1]
        
        if  np.linalg.norm( np.zeros(2) - last_trace  ) < .9: 
           print("OKAY")
        else:
           print("NOT OKAY")
        


if __name__ == "__main__":
    
    c = 3
    a=4
    problem = MathematicalProgramTraced(Phi(c = c, a=4))
    
    solver = SolverNGMethod(x_init = np.array([-1,1]))
       
    print("GAUSSEN NEWTON")
    Test(problem, solver).test_convergence()
 
    solver = SolverGradient(x_init = np.array([-1,1]))
    print("GRADIENT")
    Test(problem, solver).test_convergence()



