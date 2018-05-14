'''
Markov model main module 
'''

import numpy as np
from algs.utils import get_adjacent_state_matrix, get_next_state_probability

from wrapper import Method


class MethodMarkov(Method):
    
    def __init__(self, num_states, adjacency_list, reg_factor=1e-3):
        super(MethodMarkov, self).__init__(num_states, adjacency_list)
        self._rf = reg_factor
        
    def train(self, data):
        if not isinstance(data, list):
            raise TypeError
        
        adjacent_state_mat = get_adjacent_state_matrix(
            range(self._num_states), 
            self._adjacency_list)
        
        self._T = compute_transition_matrix(data, self._num_states, adjacent_state_mat, self._rf)
            
            
    def predict(self, sequence, *args):
        return get_next_state_probability(self._T, sequence[0][-2]), None  # last element of the sequence is the target
    
    
    def name(self):
        return 'markov'
    
    def tostr(self):
        return 'markov_rf{:.2f}'.format(self._rf)
    
            
def compute_transition_matrix(data, num_states, adjacent_state_mat, epsilon):
    '''
    Create transition matrix using training-data. 
    Input the num_of_state (400 for navigation domain, 14 for User Interface)
    and Epsilon(the regularization parameter)
    and Output the transition matrix T
    '''
    T = np.zeros((num_states, num_states))
    
    # Go over the training  data to count transition
    for i in range(len(data)):
        for j in range(len(data[i]) - 2):
            current_state = data[i][j]
            next_state = data[i][j+1]
            T[current_state][next_state] += 1
        
    #regularization
    for i in range(len(adjacent_state_mat)):
        for j in range(len(adjacent_state_mat[i])):
            if(adjacent_state_mat[i][j] == 1):
                T[i][j] += epsilon
                
    # Normalization by row
    row_sums = T.sum(axis=1)
    new_matrix = T / row_sums[:, np.newaxis]
    return new_matrix

