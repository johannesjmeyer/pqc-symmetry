import numpy as np
# import networkx as nx
import time
import scipy
import sys
from qiskit.circuit import Parameter
from qiskit.utils import QuantumInstance
from qiskit import Aer
from qiskit.opflow import CircuitStateFn
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from scipy.optimize import minimize

from ansatz_lib import makecircTTTsym, makecircTTTNOsym, makecirc_TFIM, makecirc_TFIM_RY, makecirc_TFIM_RYdiff  
from hamiltonian_lib import hamTTT,hamTFIM
from BP_lib import CostNEW, CostNEW_scipy
from myfunctions import fidel, residual, makeinit

from qiskit.algorithms.optimizers import L_BFGS_B, GradientDescent, ADAM

A = int(sys.argv[1])
B = int(sys.argv[2])
C = int(sys.argv[3])
D = int(sys.argv[4])
E = str(sys.argv[5])
F = str(sys.argv[6])
#G = int(sys.argv[7])

#Input___ MAIN Optimization
def totpar(L,P):
    if (E=="sym"):
        #return L*P + 2*P # QAOA with ry diff
        return 2*P #2*P # QAOA with ry equal
    else:
        return 3*P
        #return 2*L*P #6*P
        #return L*P # number of parameters per layer

namesym=E
if(namesym=="sym"):
    ansatznow=makecirc_TFIM
else:
    ansatznow=makecirc_TFIM_RY
hamnow=hamTFIM #hamTTT

maxfun=10**10 #1000#10**(10) # Default is 100, but 1000 for educated guess (N=8) and BP
maxiter=B #10**(10)
Lmin=D
Lmax=D
Pmin= C
Pmax= C
Pstep=2
optname=F
options={'disp': True, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': maxfun, 'maxiter': maxiter, 'iprint': 2, 'maxls': 20, 'finite_diff_rel_step': None}

#if (optname=="LBFGS"):
#    optnow=L_BFGS_B(maxfun=maxfun, maxiter=maxiter, ftol=2.220446049250313e-15, iprint=-1, eps=1e-08,  max_evals_grouped=10**10)
#else:
#    optnow=ADAM(maxiter=maxiter, tol=1e-04, lr=0.001, beta_1=0.9, beta_2=0.99, noise_factor=1e-08, eps=1e-10, amsgrad=False, snapshot_dir=None)
#ADAM(maxiter=10000, tol=1e-04, lr=0.001, beta_1=0.9, beta_2=0.99, noise_factor=1e-08, eps=1e-10, amsgrad=False, snapshot_dir=None)
#GradientDescent(maxiter=maxiter, learning_rate=0.01, tol=1e-07, callback=None, perturbation=None)
#
namefile=f"TFIM_{namesym}_{optname}_iter{maxiter}_P{Pmin}_L{Lmin}_r{A}"
#End INPUT__

# output file
outfileOpt = f"temp_results_TFIM/file_Opt"+namefile+".npz"

# for saving results
#enALL=[]
#resenALL=[]
#fidALL=[]
#optparlistALL=[]
#itertot=[]
#first_resALL=[]
#EnExact_ALL=[]
#EnMaxExact_ALL=[]

L=Lmin
print("L="+str(L)+"_________________")


H_tot = hamnow(L)
ed_result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(operator=H_tot) # exact diag. gs energy
emin = ed_result.eigenvalue.real 
psi0 = ed_result.eigenstate.to_matrix(massive=True)
#
ed_result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(operator=-H_tot)
emax= -ed_result.eigenvalue.real

EnExact_ALL=emin
EnMaxExact_ALL=emax

qi=QuantumInstance(backend=Aer.get_backend('qasm_simulator'))

P=Pmin
print("P="+str(P)+"_________________")
circ = ansatznow(P,L)
#values = []
#def callbackFun(xk): #Eventual CallBack
            #global eval_count, emin, emax, Ht
            #values.append(mean)
            #if(eval_count%100==0):
                #print(repr(np.array(xk))) 
            
            #print(str(eval_count)+": residual_en="+str(residual((n/2)*CostNEW(xk,Ht,deltaY,deltaZ,P,n),emin,emax)))    
            #eval_count=eval_count+1
init = np.random.rand(totpar(L,P))*2*np.pi
result=scipy.optimize.minimize(CostNEW_scipy, init, args=(H_tot, circ), method='L-BFGS-B', callback=None, options=options)

circ=ansatznow(P,L)
qqq=circ.assign_parameters(result.x)
psi=CircuitStateFn(qqq)

enALL=result.fun
resenALL=residual( result.fun, emin, emax)
fidALL=fidel(psi.to_matrix(massive= True),psi0)
print("FID="+str(fidALL)+"_________________")
optparlistALL=result.x
itertot=result.nfev
nittot=result.nit
#status=result.status

#print(f'Number evaluations: {result.nfev} ')
#first_res.append(residual(values[0],emin,emax))



np.savez(outfileOpt, enALL=enALL, resenALL=resenALL, fidALL=fidALL,itertot=itertot,nittot=nittot,optparlistALL=optparlistALL, EnExact_ALL=EnExact_ALL, EnMaxExact_ALL=EnMaxExact_ALL)
   


