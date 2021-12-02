import numpy as np
import sys
import warnings

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT

warnings.filterwarnings('ignore')
class SolverUnconstrained(NLPSolver):

    def __init__(self):
        """
        See also:
        ----
        NLPSolver.__init__
        """

        # in case you want to initialize some class members or so...

   #the class checks if all eigenvalues are positive. If they are not, then it increases lambda to adjust H. 
   # The programm tries 100 times and then returns a mistake. 
    def findLamda(self,H, lamda, count):
        count +=1
        if count>100:
            self.error = True
            return None,False
        try:
            if np.all(np.linalg.eigvals(H+lamda*np.identity(H.shape[0]))>0):
             return lamda, False
            if np.all(np.linalg.eigvals(H+lamda*np.identity(H.shape[0]))<0):
                return lamda, True
            else:
                return self.findLamda(H, lamda+0.01, count)
        except np.linalg.LinAlgError:
            self.error = True
            return None,False

     #calculate the step for phi or f
    def calculateDelta(self,H,grad):
        #check if all eigenvalues later then and adjust lambda
         lamda, inverse = self.findLamda(H,0,0)
         #if there was no lambda found, then the programm takes a gradient descent
         if self.error==True:
                delta = -grad/np.linalg.norm(grad)
                delta = delta.reshape((delta.shape[0],))
         else:
             if(inverse== True):
                H=-H
             try:
                 #try to get delta with lambda
                    delta = np.linalg.solve(H+lamda*np.identity(H.shape[0]), -grad)
                    delta = delta.reshape((delta.shape[0],))
             except np.linalg.LinAlgError:
                 #if it is not possible^ take a gradient descent
                        delta = -grad/np.linalg.norm(grad)
                        delta = delta.reshape((delta.shape[0],))
         return delta

    def solve(self):
        """

        See Also:
        ----
        NLPSolver.solve

        """

        # write your code here

        x = self.problem.getInitializationSample()
        types = self.problem.getFeatureTypes()
        # find indexes for f and phi problem
        index_f = [i for i, x in enumerate(types) if x == OT.f] #get all features of type f assert( len(index_f) <= 1 ) # at most, only one term of type OT.f
        index_r = [i for i, x in enumerate(types) if x == OT.sos] #get all sum-of-square features phi, J = problem.evaluate(x)
        
        delta = [1]
        alpha=1  
        count =0
        while np.linalg.norm(delta*alpha)>0.0001 or count >1000:
            count +=1
            grad = 0 
            H= 0
            c=0
            phi, J = self.problem.evaluate(x)
            delta =0
            self.error = False
            # divide the problem and find approximate the step for phi and f
            if len(index_f) > 0 :
                c += phi[index_f][0]
                grad +=J[index_f].T
                grad = np.reshape(grad,(grad.shape[0],))
                H += self.problem.getFHessian(x)
                delta += self.calculateDelta(H,grad)
            if len(index_r) > 0 :
                c += phi[index_r].T @ phi[index_r]
                grad += J[index_r].T@phi[index_r]
                grad = np.reshape(grad,(grad.shape[0],))
                H += 2*J[index_r].T@J[index_r]
                delta += self.calculateDelta(H,grad)
            #line-search. We shold check if the Wolf conditiion is satisfied and choose a proper alpha.
            n_phi= self.problem.evaluate(x+alpha*delta)[0]
            n_c =0
            if len(index_f) > 0 :
                 n_c += n_phi[index_f][0]
            if len(index_r) > 0 :
                n_c += n_phi[index_r].T @ n_phi[index_r]
            compare = c+ 0.01* grad.T @(alpha*delta)
            ro_minus = 0.5
            while (n_c > compare).all():
                n_c =0
                alpha = ro_minus*alpha
                n_phi= self.problem.evaluate(x+alpha*delta)[0]
                if len(index_f) > 0 :
                 n_c += n_phi[index_f][0]
                if len(index_r) > 0 :
                 n_c += n_phi[index_r].T @ n_phi[index_r]
                compare = c + 0.01* grad.T @(alpha*delta)
                delta = alpha * delta
                
            #update x and alpha
            x = x+delta*alpha
            alpha = min(1.2*alpha,1)
        #print the amount of iterations and the problem.
        print(self.problem.report(1), ":\n iterations: " ,count) 
        return x
