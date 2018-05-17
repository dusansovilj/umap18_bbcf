'''
Markov model main module 
'''

import numpy as np
from math import log
from algs.utils import get_adjacent_state_matrix, get_next_state_probability,\
    get_start_state_distribution

from wrapper import Method


class MethodMarkov(Method):
    
    def __init__(self, num_states, adjacency_list, reg_factor=1e-3):
        super(MethodMarkov, self).__init__(num_states, adjacency_list)
        self._rf = reg_factor
        
    def train(self, data):
        adjacent_state_mat = get_adjacent_state_matrix(
            range(self._num_states), 
            self._adjacency_list)
        
        self._T = compute_transition_matrix(
            data, 
            self._num_states, 
            adjacent_state_mat, 
            self._rf)
        
        self._init_distribution = np.zeros((self._num_states,), dtype='float32') + self._rf
        for sample in data:
            self._init_distribution[sample[0]] += 1. 
        self._init_distribution /= len(data)
            
            
    def predict(self, sequence, *args):
        assert len(sequence[0]) > 1
        return get_next_state_probability(self._T, sequence[0][-2]), None  # last element of the sequence is the target
    
    
    def loglike(self, sequence, *args):
        '''
        Compute log-likelihood of a sequence under this model. 
        If additional argument is given, the function 
        interprets it as log-likelihood computed from the same
        sequence but with last state removed, that is, 
        loglike(sequence[:-1]). 
        '''
        
        assert len(sequence) > 0
        
        if args[0] is None:
            loglike = log(self._init_distribution[sequence[0]])
            for s in range(1, len(sequence)):
                ap = self._T[sequence[s-1]]
                pr = ap[sequence[s]]
                loglike += log(pr) if pr > 0 else log(self._rf)
        else:
            loglike = args[0]
            ap = self._T[sequence[-2]]
            pr = ap[sequence[-1]]
            loglike += log(pr) if pr > 0 else log(self._rf)
            
        return loglike 
        
    
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

