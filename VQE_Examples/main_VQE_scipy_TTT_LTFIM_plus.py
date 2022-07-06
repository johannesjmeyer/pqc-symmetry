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

from ansatz_lib import makecirc_TTTLTFIM_nosym_plus ,makecirc_TTTLTFIM_sym_plus
from hamiltonian_lib import hamTTT_TFIM
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
        return 9*P #2*P # QAOA with ry equal
    else:
        return 2*L*P + 16*P
        #return 2*L*P #6*P
        #return L*P # number of parameters per layer

namesym=E

listCorn=[0,2,6,8]
listBord=[1,3,5,7]
listCent=[4]
edgCont=[]
edgCont.append((0,1))
edgCont.append((1,2))
edgCont.append((2,5))
edgCont.append((5,8))
edgCont.append((7,8))
edgCont.append((6,7))
edgCont.append((0,3))
edgCont.append((3,6))

edgIns=[]
edgIns.append((1,4))
edgIns.append((3,4))
edgIns.append((4,5))
edgIns.append((4,7))

edgDiag=[]
edgDiag.append((0,4))
edgDiag.append((2,4))
edgDiag.append((6,4))
edgDiag.append((8,4))

def ansatznow(P,L):
    if(namesym=="sym"):
        parA_X=[]
        parB_X=[]
        parC_X=[]
        parA_ZZ=[]
        parB_ZZ=[]
        parC_ZZ=[]
        parA_Z=[]
        parB_Z=[]
        parC_Z=[]
        for i in range(P):
            parA_X.append(Parameter(f'ax{i:013b}'))
            parB_X.append(Parameter(f'bx{i:013b}'))
            parC_X.append(Parameter(f'cx{i:013b}'))
            parA_ZZ.append(Parameter(f'azz{i:013b}'))
            parB_ZZ.append(Parameter(f'bzz{i:013b}'))
            parC_ZZ.append(Parameter(f'czz{i:013b}'))
            parA_Z.append(Parameter(f'az{i:013b}'))
            parB_Z.append(Parameter(f'bz{i:013b}'))
            parC_Z.append(Parameter(f'cz{i:013b}'))
        return makecirc_TTTLTFIM_sym_plus(parA_X,parB_X,parC_X,parA_ZZ,parB_ZZ,parC_ZZ,parA_Z,parB_Z,parC_Z,P,L,listCorn,listBord,listCent,edgCont,edgIns,edgDiag)
    else:
        parA=[]
        parB_cont=[]
        parB_ins=[]
        parB_diag=[]
        parC=[]

        for i in range(P*L):
            parA.append(Parameter(f'a{i:013b}'))
            parC.append(Parameter(f'c{i:013b}'))

        for i in range(len(edgCont)*P):
            parB_cont.append(Parameter(f'bc{i:013b}'))

        for i in range(len(edgIns)*P):
            parB_ins.append(Parameter(f'bi{i:013b}'))

        for i in range(len(edgDiag)*P):
            parB_diag.append(Parameter(f'bd{i:013b}'))
        return makecirc_TTTLTFIM_nosym_plus(parA,parB_cont,parB_ins,parB_diag,parC,P,L,edgCont,edgIns,edgDiag)





hamnow=hamTTT_TFIM #hamTTT

maxfun=10**10 #1000#10**(10) 
maxiter=B #10**(10)
Pmin= C
Pmax= C
Lmin=D
Lmax=D
optname=F
options={'disp': True, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': maxfun, 'maxiter': maxiter, 'iprint': 2, 'maxls': 20, 'finite_diff_rel_step': None}

namefile=f"TTTLTFIMplus_{namesym}_{optname}_iter{maxiter}_P{Pmin}_L{Lmin}_r{A}"
#End INPUT__

# output file
outfileOpt = f"temp_results_TTTLTFIM/file_Opt"+namefile+".npz"

 
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
   


