import numpy as np
from qiskit.opflow import *

"""
Build an interaction term as XX, YY, ZZ
""" 

def termXX(i,j,n):
    i=n-i-1
    j=n-j-1
    for pos in range(n):
        if pos==0:
            if(pos==i or pos==j):
                ter=X
            else:
                ter=I
        else:        
            if(pos==i or pos==j):
                ter=ter^X
            else:
                ter=ter^I
    return ter

def termYY(i,j,n):
    i=n-i-1
    j=n-j-1
    for pos in range(n):
        if pos==0:
            if(pos==i or pos==j):
                ter=Y
            else:
                ter=I
        else:        
            if(pos==i or pos==j):
                ter=ter^Y
            else:
                ter=ter^I
    return ter

def termZZ(i,j,n):
    i=n-i-1
    j=n-j-1
    for pos in range(n):
        if pos==0:
            if(pos==i or pos==j):
                ter=Z
            else:
                ter=I
        else:        
            if(pos==i or pos==j):
                ter=ter^Z
            else:
                ter=ter^I
    return ter


"""
Build the interaction Hamiltonian, given a weighted graph
"""

def makeXX(edges_list,coeff_list,n):
    Hnow=0
    for (edge,w) in zip(edges_list,coeff_list):
        i=edge[0]
        j=edge[1]
        Hnow = Hnow + w*(termXX(i,j,n))
    return Hnow

def makeYY(edges_list,coeff_list,n):
    Hnow=0
    for (edge,w) in zip(edges_list,coeff_list):
        i=edge[0]
        j=edge[1]
        Hnow = Hnow + w*(termYY(i,j,n))
    return Hnow

def makeZZ(edges_list,coeff_list,n):
    Hnow=0
    for (edge,w) in zip(edges_list,coeff_list):
        i=edge[0]
        j=edge[1]
        Hnow = Hnow + w*(termZZ(i,j,n))
    return Hnow


"""
Build the one-body Hx Hamiltonian
"""

def termX(i,n):
    i=n-i-1
    te=1
    for pos in range(n):
        if pos==0:
            if(pos==i):
                te=X
            else:
                te=I
        else:        
            if(pos==i):
                te=te^X
            else:
                te=te^I
    return te

def termZ(i,n):
    i=n-i-1
    te=1
    for pos in range(n):
        if pos==0:
            if(pos==i):
                te=Z
            else:
                te=I
        else:        
            if(pos==i):
                te=te^Z
            else:
                te=te^I
    return te

def makeX(n): # = sum_i X_i
    Hxnow=1.0*(termX(0,n))
    for i in range(1,n):
        Hxnow = Hxnow + 1.0*(termX(i,n))
    return Hxnow

def makeZ(n): # = sum_i Z_i
    Hxnow=1.0*(termZ(0,n))
    for i in range(1,n):
        Hxnow = Hxnow + 1.0*(termZ(i,n))
    return Hxnow

"""
Create only interaction terms on the first even AND odd links (we shall use translational invariance)
"""

def makeZZfirstEO(n): 
    Hnow = 1*(termZZ(0,1,n)) + 1*(termZZ(1,2,n))
    return Hnow

def makeYYfirstEO(n):
    Hnow = 1*(termYY(0,1,n)) + 1*(termYY(1,2,n))
    return Hnow

def makeXXfirstEO(n):
    Hnow = 1*(termXX(0,1,n)) + 1*(termXX(1,2,n))
    return Hnow


# TFIM

def makeZZfirst_TFIM(n): # only pair (0,1)
    Hnow = 1*(termZZ(0,1,n))
    return Hnow

def makeXfirst(n): # only qubit #0
    Hnow = 1*(termX(0,n))
    return Hnow



def hamTTT(L):
    edges=[]
    edges.append((0,1))
    edges.append((1,2))
    edges.append((3,4))
    edges.append((4,5))
    edges.append((6,7))
    edges.append((7,8))

    for i in range(6):
        edges.append((i,(i+3)%L))

    edges.append((0,4))
    edges.append((2,4))
    edges.append((6,4))
    edges.append((8,4))

    coeff=np.ones(len(edges))
    diag=2
    coeff[len(edges)-4]=diag
    coeff[len(edges)-3]=diag
    coeff[len(edges)-2]=diag
    coeff[len(edges)-1]=diag


    Hzz = makeZZ(edges,coeff,L)
    #Hx = makeX(nqubits)
    #Hyy = makeYY(edges,coeff,L)
    #Hxx = makeXX(edges,coeff,L)
    #Hz = makeZ(L)
    Hx = makeX(L)
    #Ht = 1*Hzz +1*Hyy + Hxx + Hz + 0.1*termZ(i=4,n=9)
    Ht= Hzz + Hx + 0.1*termX(i=4,n=9)
    return Ht


def hamTFIM(L):
    edges=[]
    for i in range(L):
        edges.append((i,(i+1)%L))
    coeff=np.ones(len(edges))
    Hzz = makeZZ(edges,coeff,L)
    #Hyy = makeYY(edges,coeff,L)
    #Hxx = makeXX(edges,coeff,L)
    Hx = makeX(L)
    Ht = -1*Hzz - Hx 
    return Ht


def hamXXX(L):
    edges=[]
    for i in range(L):
        edges.append((i,(i+1)%L))
    coeff=np.ones(len(edges))
    Hzz = makeZZ(edges,coeff,L)
    Hyy = makeYY(edges,coeff,L)
    Hxx = makeXX(edges,coeff,L)
    #Hx = makeX(L)
    Ht = Hxx + Hyy + Hzz 
    return Ht


def hamTTT_TFIM(L):
    edgCont=[]
    edgCont.append((0,1))
    edgCont.append((1,2))
    edgCont.append((2,5))
    edgCont.append((5,8))
    edgCont.append((7,8))
    edgCont.append((6,7))
    edgCont.append((0,3))
    edgCont.append((3,6))
    coeffCont=np.ones(len(edgCont))

    edgIns=[]
    edgIns.append((1,4))
    edgIns.append((3,4))
    edgIns.append((4,5))
    edgIns.append((4,7))
    coeffIns=1.5*np.ones(len(edgIns))

    edgDiag=[]
    edgDiag.append((0,4))
    edgDiag.append((2,4))
    edgDiag.append((6,4))
    edgDiag.append((8,4))
    coeffDiag=0.5*np.ones(len(edgDiag))


    Hzz = makeZZ(edgCont,coeffCont,L) + makeZZ(edgIns,coeffIns,L) + makeZZ(edgDiag,coeffDiag,L)
    Hz = makeZ(L)
    Hx = makeX(L)
    Ht = 1*Hzz +1*Hx + 1*Hz 
    return Ht
