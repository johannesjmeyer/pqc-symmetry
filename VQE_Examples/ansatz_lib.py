import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter




"""
Create the ansatz circuit for the TFIM model
"""
def makecirc_TFIM(PP,dim):

    parGam=[]
    parBet=[]
    for i in range(PP):
        parGam.append(Parameter(f'a{i:010b}'))
        parBet.append(Parameter(f'b{i:010b}'))
    
    ci = QuantumCircuit(dim)
    for i in range(dim):
        ci.h(i)
    #ci.barrier()
    for p in range(PP):
        for i in range(dim//2):
            ci.rzz(-2*parGam[p],2*i,2*i+1)
        for i in range(dim//2):
            ci.rzz(-2*parGam[p],2*i+1,(2*i+2)%dim)
        if (dim%2==1):
            ci.rzz(-2*parGam[p],dim-1,0)
        #circ.barrier()
        for ii in range(dim):
            ci.rx(-2*parBet[p],ii)     
            
    return ci        


def makecirc_TFIM_RY(PP,dim):
    
    parA=[]
    parB=[]
    parC=[]
    for i in range(PP):
        parA.append(Parameter(f'a{i:010b}'))
        parB.append(Parameter(f'b{i:010b}'))
        parC.append(Parameter(f'c{i:010b}'))
    
    ci = QuantumCircuit(dim)
    for i in range(dim):
        ci.h(i)
    #ci.barrier()
    for p in range(PP):
        for i in range(dim//2):
            ci.rzz(-2*parA[p],2*i,2*i+1)
        for i in range(dim//2):
            ci.rzz(-2*parA[p],2*i+1,(2*i+2)%dim)
        if (dim%2==1):
            ci.rzz(-2*parA[p],dim-1,0)
        #circ.barrier()
        for ii in range(dim):
            ci.rx(-2*parB[p],ii)    
        for ii in range(dim):
            ci.ry(-2*parC[p],ii)     
            
    return ci        


def makecirc_TFIM_RYdiff(PP,dim):
    parA=[]
    parB=[]
    parC=[]
    for i in range(PP):
        parA.append(Parameter(f'a{i:010b}'))
        parB.append(Parameter(f'b{i:010b}'))
    
    for i in range(dim*PP):
        parC.append(Parameter(f'c{i:010b}'))
    
    ci = QuantumCircuit(dim)
    for i in range(dim):
        ci.h(i)
    #ci.barrier()
    for p in range(PP):
        for i in range(dim//2):
            ci.rzz(-2*parA[p],2*i,2*i+1)
        for i in range(dim//2):
            ci.rzz(-2*parA[p],2*i+1,(2*i+2)%dim)
        if (dim%2==1):
            ci.rzz(-2*parA[p],dim-1,0)
        #circ.barrier()
        for ii in range(dim):
            ci.rx(-2*parB[p],ii)    
        for ii in range(dim):
            ci.ry(-2*parC[p*dim + ii],ii)     
    return ci    




def makecircXXX(PP,dim):
    
    # included this in the function
    parA=[]
    parB=[]
    '''
    parC=[]
    parD=[]
    parE=[]
    parF=[]
    '''
    for i in range(PP):
        parA.append(Parameter(f'a{i:010b}'))
        parB.append(Parameter(f'b{i:010b}'))
        '''
        parC.append(Parameter(f'c{i:010b}'))
        parD.append(Parameter(f'd{i:010b}'))
        parE.append(Parameter(f'e{i:010b}'))
        parF.append(Parameter(f'f{i:010b}'))
        '''
    ci = QuantumCircuit(dim)    
    for i in range(dim):
        ci.x(i)
    for i in range(dim):    
        if (i%2==0):
            ci.h(i)
            ci.cx(i,(i+1)%dim)
    for p in range(PP):
        for i in range((dim)//2):
            ci.rzz(2*parA[p],(2*i+1)%dim,(2*i+2)%dim)
        for i in range((dim)//2):
            ci.ryy(2*parA[p],(2*i+1)%dim,(2*i+2)%dim)
        for i in range((dim)//2):
            ci.rxx(2*parA[p],(2*i+1)%dim,(2*i+2)%dim)
        for i in range((dim+1)//2):
            ci.rzz(2*parB[p],(2*i)%dim,(2*i+1)%dim)
        for i in range((dim+1)//2):
            ci.ryy(2*parB[p],(2*i)%dim,(2*i+1)%dim)
        for i in range((dim+1)//2):
            ci.rxx(2*parB[p],(2*i)%dim,(2*i+1)%dim)
    return ci
        
        
def makecircXYZ(PP,dim):
    
    # included this in the function
    parA=[]
    parB=[]
    parC=[]
    parD=[]
    parE=[]
    parF=[]
    
    for i in range(PP):
        parA.append(Parameter(f'a{i:010b}'))
        parB.append(Parameter(f'b{i:010b}'))
        parC.append(Parameter(f'c{i:010b}'))
        parD.append(Parameter(f'd{i:010b}'))
        parE.append(Parameter(f'e{i:010b}'))
        parF.append(Parameter(f'f{i:010b}'))
        
    ci = QuantumCircuit(dim)    
    for i in range(dim):
        ci.x(i)
    for i in range(dim):    
        if (i%2==0):
            ci.h(i)
            ci.cx(i,(i+1)%dim)
    for p in range(PP):
        for i in range((dim)//2):
            ci.rzz(2*parA[p],(2*i+1)%dim,(2*i+2)%dim)
        for i in range((dim)//2):
            ci.ryy(2*parB[p],(2*i+1)%dim,(2*i+2)%dim)
        for i in range((dim)//2):
            ci.rxx(2*parC[p],(2*i+1)%dim,(2*i+2)%dim)
        for i in range((dim+1)//2):
            ci.rzz(2*parD[p],(2*i)%dim,(2*i+1)%dim)
        for i in range((dim+1)//2):
            ci.ryy(2*parE[p],(2*i)%dim,(2*i+1)%dim)
        for i in range((dim+1)//2):
            ci.rxx(2*parF[p],(2*i)%dim,(2*i+1)%dim)
    return ci
                
                
                
                
                
                
                
def makecircXYZ_RY(PP,dim):
    
    # included this in the function
    parA=[]
    parB=[]
    parC=[]
    parD=[]
    parE=[]
    parF=[]
    parG=[]
    
    for i in range(PP):
        parA.append(Parameter(f'a{i:010b}'))
        parB.append(Parameter(f'b{i:010b}'))
        parC.append(Parameter(f'c{i:010b}'))
        parD.append(Parameter(f'd{i:010b}'))
        parE.append(Parameter(f'e{i:010b}'))
        parF.append(Parameter(f'f{i:010b}'))
        parG.append(Parameter(f'g{i:010b}'))
        
    ci = QuantumCircuit(dim)    
    for i in range(dim):
        ci.x(i)
    for i in range(dim):    
        if (i%2==0):
            ci.h(i)
            ci.cx(i,(i+1)%dim)
    for p in range(PP):
        for i in range((dim)//2):
            ci.rzz(2*parA[p],(2*i+1)%dim,(2*i+2)%dim)
        for i in range((dim)//2):
            ci.ryy(2*parB[p],(2*i+1)%dim,(2*i+2)%dim)
        for i in range((dim)//2):
            ci.rxx(2*parC[p],(2*i+1)%dim,(2*i+2)%dim)
        for i in range((dim+1)//2):
            ci.rzz(2*parD[p],(2*i)%dim,(2*i+1)%dim)
        for i in range((dim+1)//2):
            ci.ryy(2*parE[p],(2*i)%dim,(2*i+1)%dim)
        for i in range((dim+1)//2):
            ci.rxx(2*parF[p],(2*i)%dim,(2*i+1)%dim)
        for i in range(dim):
            ci.ry(2*parG[p],i)
    return ci
                                


                

def  makecirc_TTTLTFIM_sym(parA_X,parB_X,parC_X,parA_ZZ,parB_ZZ,parC_ZZ,parA_Z,parB_Z,parC_Z,PP,dim,listCorn,listBord,listCent,edgCont,edgIns,edgDiag):
    
    ci = QuantumCircuit(dim)
    for p in range(PP):
        for i in listCorn:
            ci.rx(parA_X[p],i)
        for i in listBord:
            ci.rx(parB_X[p],i)
        for i in listCent:
            ci.rx(parC_X[p],i)
        
        for edge in edgCont:
            ci.rzz(2*parA_ZZ[p],edge[0],edge[1])
        
        for edge in edgIns:
            ci.rzz(2*parB_ZZ[p],edge[0],edge[1])
            
        for edge in edgDiag:
            ci.rzz(2*parC_ZZ[p],edge[0],edge[1])

        for i in listCorn:
            ci.rz(parA_Z[p],i)
        for i in listBord:
            ci.rz(parB_Z[p],i)
        for i in listCent:
            ci.rz(parC_Z[p],i)  

    return ci




def makecirc_TTTLTFIM_nosym(parA,parB_cont,parB_ins,parB_diag,parC,PP,dim,edgCont,edgIns,edgDiag):
    
    ci = QuantumCircuit(dim)
    for p in range(PP):
        for i in range(dim):
            ci.rx(parA[p*dim + i],i)
        
        for i,edge in enumerate(edgCont):
            ci.rzz(2*parB_cont[p*len(edgCont)+i],edge[0],edge[1])
        
        for i,edge in enumerate(edgIns):
            ci.rzz(2*parB_ins[p*len(edgIns)+i],edge[0],edge[1])
            
        for i,edge in enumerate(edgIns):
            ci.rzz(2*parB_diag[p*len(edgDiag)+i],edge[0],edge[1])

        for i in range(dim):
            ci.rz(parC[p*dim + i],i)
     
    return ci




def  makecirc_TTTLTFIM_sym_plus(parA_X,parB_X,parC_X,parA_ZZ,parB_ZZ,parC_ZZ,parA_Z,parB_Z,parC_Z,PP,dim,listCorn,listBord,listCent,edgCont,edgIns,edgDiag):
    
    ci = QuantumCircuit(dim)
    for i in range(dim):
        ci.h(i)
    for p in range(PP):
        for edge in edgCont:
            ci.rzz(2*parA_ZZ[p],edge[0],edge[1])
        
        for edge in edgIns:
            ci.rzz(2*parB_ZZ[p],edge[0],edge[1])
            
        for edge in edgDiag:
            ci.rzz(2*parC_ZZ[p],edge[0],edge[1])

        for i in listCorn:
            ci.rx(parA_X[p],i)
        for i in listBord:
            ci.rx(parB_X[p],i)
        for i in listCent:
            ci.rx(parC_X[p],i)

        for i in listCorn:
            ci.rz(parA_Z[p],i)
        for i in listBord:
            ci.rz(parB_Z[p],i)
        for i in listCent:
            ci.rz(parC_Z[p],i)  

    return ci




def makecirc_TTTLTFIM_nosym_plus(parA,parB_cont,parB_ins,parB_diag,parC,PP,dim,edgCont,edgIns,edgDiag):
    
    ci = QuantumCircuit(dim)
    for i in range(dim):
        ci.h(i)
    for p in range(PP):
        
        for i,edge in enumerate(edgCont):
            ci.rzz(2*parB_cont[p*len(edgCont)+i],edge[0],edge[1])
        
        for i,edge in enumerate(edgIns):
            ci.rzz(2*parB_ins[p*len(edgIns)+i],edge[0],edge[1])
            
        for i,edge in enumerate(edgIns):
            ci.rzz(2*parB_diag[p*len(edgDiag)+i],edge[0],edge[1])
        
        for i in range(dim):
            ci.rx(parA[p*dim + i],i)
        
        for i in range(dim):
            ci.rz(parC[p*dim + i],i)
     
    return ci
