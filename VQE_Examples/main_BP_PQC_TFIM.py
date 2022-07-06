import numpy as np
import os
# import matplotlib.pyplot as plt
import time
import sys
from qiskit.circuit import Parameter

from ansatz_lib import makecirc_TFIM, makecirc_TFIM_RY
from hamiltonian_lib import termZZ
from BP_lib import CostNEW



# rand input parameters
nrands=int(sys.argv[1]) # 1000 # random sampling points for gradient estimation
L=int(sys.argv[2]) 
P=int(sys.argv[3])
ans=str(sys.argv[4])
epsd= 0.000001 #epsilon of *finite difference gradient* estimation 
#

if(ans=="sym"):
    prep=2
else:
    prep=3

def ansatz(P,L):
    if(ans=="sym"):
        return makecirc_TFIM(P,L,)
    else:
        return makecirc_TFIM_RY(P,L) 


Ht_BP = termZZ(0,1,L)
##### 1) BP Random points
# output file
outfile_BP_rand = f"temp_results_TFIM/fileBP_TFIM_rand_L={L}_P={P}_{ans}.npz"

# circuit for given N and P
circ_BP=ansatz(P,L)         
sumcost=0
Grad_sample=np.zeros(nrands)
Grad_sample_sq=np.zeros(nrands)

for i in range(nrands): #for each i, compute the partial derivatives and the cost function in a random point parR
    parR= np.random.rand(prep*P)*np.pi*2
    # only one partial derivative, wrt first component
    pardx= np.copy(parR)
    pardx[0] = pardx[0] + epsd
    parsx= np.copy(parR)
    parsx[0]= parsx[0] - epsd
    #first component of the gradient: finite differences
    grad = (CostNEW(pardx, Ht_BP, circ_BP) - CostNEW(parsx, Ht_BP, circ_BP)) / (2*epsd) 
    #
    cost_now=CostNEW(parR, Ht_BP, circ_BP) #cost function evaluated at the point parR
    #print("Cost= "+str(cost_now))
    sumcost+= cost_now
    Grad_sample[i]=grad
    Grad_sample_sq[i]= grad**2
    #Grad_sample.append(grad) 

Meancost = sumcost/nrands
Meangrad = np.mean(Grad_sample)
Vargrad =  np.var(Grad_sample,ddof=1)
Mean_grad_sq= np.mean(Grad_sample_sq)
Mean_grad_abs=np.mean(np.abs(Grad_sample))
print(f"\nL={L}\tP={P}")
print(f"Vargrad = {Vargrad}\n")

np.savez(outfile_BP_rand, Vargrad=Vargrad, Meancost=Meancost, Meangrad=Meangrad, Mean_grad_sq=Mean_grad_sq, Mean_grad_abs=Mean_grad_abs)
        
