import numpy as np
import sys
import math

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT


class SolverAugmentedLagrangian(NLPSolver):

   

    def setProblem(self,problem):
        self.problem = problem

    def calculate_uneq_contrains(self,phi,J,index_in,mu, la):
        g = phi[index_in]
        dg = J[index_in].T
        B=mu*((~((g<0)*( la<=0))*(g**2)).sum()) + la.T@g
        if (g>0).any():
            dB = mu*( (~((g<0)*( la<=0))*2*g*dg).sum(axis=1)) + (la*dg).sum(axis=1)
        else: 
            dB =0
        if (g>0).any():
            ddB = mu*((~((g<0)*( la<=0)))*2*(dg))@(dg.T) 
        else: 
            ddB =0 
        return B, dB,ddB

    def calculate_eq_contrains(self,phi,J,index_eq,mu, ka):
        h = phi[index_eq]
        dh = J[index_eq].T
        B=mu*((h**2).sum()) + ka.T@h
        dB = mu*((2*h*dh).sum(axis=1)) + (ka*dh).sum(axis=1)
        ddB = mu*2*(dh@(dh.T)) 
        return B, dB,ddB
   
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
                return self.findLamda(H, lamda+0.1, count)
        except np.linalg.LinAlgError:
            self.error = True
            return None,False

     #calculate the step for phi or f
    def calculateDelta(self,H,grad):
        #check if all eigenvalues later then and adjust lambda
         lamda, inverse = self.findLamda(H,0,0)
         #if there was no lambda found, then the programm takes a gradient descent
         if self.error==True:
                print("problem 1")
                delta = -grad/np.linalg.norm(grad)
                delta = delta.reshape((delta.shape[0],))
         else:
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
        index_in = [i for i, x in enumerate(types) if x == OT.ineq] 
        index_eq = [i for i, x in enumerate(types) if x == OT.eq] 
        count =0
        mu =1
        v = 1
        phi, J = self.problem.evaluate(x)
        delta =[10]
        alpha = 1
        n_x =np.ones(*x.shape)
        n_x *=100
        while np.linalg.norm(n_x-x)>0.001 or ((phi[index_in]>0.001).all()) or ((np.linalg.norm(phi[index_eq])>0.001).all()):
            n_x = x
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
            if len(index_r) > 0 :
                c += phi[index_r].T @ phi[index_r]
                grad += J[index_r].T@phi[index_r]
                grad = np.reshape(grad,(grad.shape[0],))
                H += 2*J[index_r].T@J[index_r]
            if len(index_in)>0:
                la = np.zeros(*phi[index_in].shape)
                con_phi, con_grad, const_H = self.calculate_uneq_contrains(phi,J,index_in,mu,la)
                c += con_phi
                grad += con_grad
                grad = np.reshape(grad,(grad.shape[0],))
                H +=const_H
            if len(index_eq)>0:
                ka = np.zeros(*phi[index_eq].shape)
                con_phi, con_grad, const_H = self.calculate_eq_contrains(phi,J,index_eq,v,ka)
                c += con_phi
                grad += con_grad
                grad = np.reshape(grad,(grad.shape[0],))
                H +=const_H

            delta = self.calculateDelta(H,grad)
            
            #line-search. We shold check if the Wolf conditiion is satisfied and choose a proper alpha.
            n_phi= self.problem.evaluate(x+alpha*delta)[0]
            n_c =0
            if len(index_f) > 0 :
                n_c += n_phi[index_f][0]
            if len(index_r) > 0 :
                n_c += n_phi[index_r].T @ n_phi[index_r]
            if len(index_in)>0:
                n_con_phi= self.calculate_uneq_contrains(n_phi,J,index_in,mu,la)[0]
                n_c += n_con_phi
            if len(index_eq)>0:
                n_con_phi= self.calculate_eq_contrains(n_phi,J,index_eq,v,ka)[0]
                n_c += n_con_phi
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
                if len(index_in)>0:
                    n_con_phi = self.calculate_uneq_contrains(n_phi,J,index_in,mu,la)[0]
                    n_c += n_con_phi
                if len(index_eq)>0:
                    n_con_phi= self.calculate_eq_contrains(n_phi,J,index_eq,v,ka)[0]
                    n_c += n_con_phi
                compare = c + 0.01* grad.T @(alpha*delta)
                delta = alpha * delta
                
            #update x and alpha
            x = x+delta*alpha
            alpha = min(alpha*1.2,1)
            phi= self.problem.evaluate(x)[0]
            if len(index_in)>0:
                g = phi[index_in]
                la = (la +2*mu * g)*((la+2*mu * g)>0)
            if len(index_eq)>0:
                h = phi[index_eq]
                ka = ka+2*v*h
            mu = mu*1.2
            v = v*1.2

        #print the amount of iterations and the problem.
        print(x, ":\n iterations: " ,count) 
        return x

