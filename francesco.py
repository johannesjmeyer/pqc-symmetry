import pennylane as qml
from pennylane import numpy as np

ttt_dev = qml.device("default.qubit", wires=9) # the device used to label ttt instances

###################################################
###################################################
###################################################

def data_encoding(game):
    '''
    loops through game array, applies RX(game[i]) on wire i
    input: 9x9 np array representation of ttt game
    output: None?
    '''
    fgame = game.flatten()
    for j in range(len(fgame)):
        qml.RX(fgame[j], wires=j)


def row_layer(params):
    '''
    entangles nearest neighbours qubits on each ROW of the game through CZs depending on params
    input: 6-elements parameters vector
    '''
    for row in range(3):
        for col in range(2):
            qml.CRZ(params[2*row+col],wires=[3*row+col,3*row+col+1])
        

def column_layer(params):
    '''
    entangles nearest neighbours qubits on each COLUMN of the game through CZs depending on params
    input: 6-elements parameters vector
    '''
    for col in range(3):
        for row in range(2):
            qml.CRZ(params[2*row+col],wires=[3*col+row,3*col+row+1])





@qml.qnode(ttt_dev)
def full_circ(game, params):
        '''
        prepares the all zero comp basis state then iterates through encoding and layers
        input: params, np array of shape r x 2 x 6 
        '''
        # TODO: this should automatically start from the all-zero state in comp basis right?

        ngame = np.pi*0.5*game # normalize entries of game so they are between -pi/2, pi/2

        print('hey $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')


        for r in range(params.shape[0]): # for each of the r repetitions interleave data_enc with row and colum n layers
            print('r {}'.format(r))
            data_encoding(ngame) 
            #drawer = qml.draw(data_encoding)
            #print(drawer(ngame))

            row_layer(params[r,0])
            #drawer = qml.draw(row_layer)
            #print(drawer(params[r,0]))

            data_encoding(ngame)
            column_layer(params[r,1])

        return qml.expval(qml.PauliZ(3)) # measure everything in comp basis



###################################################
###################################################
###################################################


game = np.array([[-1, 1, 1], [0, -1, 1], [-1, 0, -1]]) # just a random game

params = np.random.uniform(low=-1, high=1, size=(5,2,6)) # random set of starting params

#print([params])



c = full_circ(game, params)

drawer = qml.draw(full_circ)
#TODO drawing fails...what am I missing?
print('yo')
print(drawer(game, params))