import numpy as np
from qiskit import Aer
from qiskit.opflow import AerPauliExpectation, CircuitSampler, StateFn, CircuitStateFn

def CostNEW(par, Ht, circNEW): 
    """
    Compute the value of the cost function at parameters par
    NEW: Ht and circNEW provided from the main
    OLD: *args = Ht,deltaY,deltaZ,P,n are the fixed variables
    """
    #
    ''' OLD
    Ht=args[0]
    deltaY=args[1]
    deltaZ=args[2]
    P=args[3]
    n=args[4]
    '''
    # MODIFICA
    qqqNEW=circNEW.assign_parameters(par)
    # Choose backend
    backend = Aer.get_backend('statevector_simulator') 
    # Create a quantum state from the QC output
    psi = CircuitStateFn(qqqNEW)
    # "Apply Ht to psi"
    measurable_expression = StateFn(Ht, is_measurement=True).compose(psi) 
    # Compute the exp val using Qiskit
    expectation = AerPauliExpectation().convert(measurable_expression)
    sampler = CircuitSampler(backend).convert(expectation) 
    temp=(sampler.eval().real)
    return temp



def CostNEW_scipy(par,*args): #par is the vector of parameters (to optimize eventually), *args=Ht,deltaY,deltaZ,P,n are the fixed variables
    Ht=args[0]
    circNEW=args[1]
    qqqNEW=circNEW.assign_parameters(par)
    psi = CircuitStateFn(qqqNEW)
    backend = Aer.get_backend('statevector_simulator') 
    measurable_expression = StateFn(Ht, is_measurement=True).compose(psi) 
    expectation = AerPauliExpectation().convert(measurable_expression)
    sampler = CircuitSampler(backend).convert(expectation) 
    temp=(sampler.eval().real)
    return temp
