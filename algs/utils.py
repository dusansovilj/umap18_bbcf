'''
Utility functions for sequence data
'''

import sys
import psutil
import numpy as np


def get_next_state_probability(T, current_state):
    '''Given the current-state and the Transition matrix, return an array of the next_state probability distribution'''
    return T[current_state]


def get_goal_state_distribution(data):
    '''Compute distribution for destination/final state in the data'''

    goal_record = dict()
    for i in range(len(data)):
        goal_state = data[i][-1]
        if goal_state in goal_record:
            goal_record[goal_state] += 1
        else:
            goal_record[goal_state] = 1
    return goal_record



def get_adjacent_state_matrix(states, adjacency=None):
    '''
    Returns an adjacency matrix based on the number of states and  
    a list of possible state transitions.
    '''
    if isinstance(adjacency, list) + isinstance(adjacency, dict) == 0:
        raise TypeError('adjacency parameter must be a list or a dictionary.')
    
    num_states = len(states)
        
    def check_values(l):
        v = [not val in states  for val in l]
        if sum(v) > 0:
            raise ValueError('State index out of bounds for list {:s}'.format(str(l)))
        
    if isinstance(adjacency, list):
        assert num_states == len(adjacency)
#         if num_states != len(adjacency):
#             print('Length of adjacency list must be equal to the number of states')
#             sys.exit()
            
        adj_matrix = np.zeros((num_states, num_states))
        for i in xrange(num_states):
            check_values(adjacency[i])
            adj_matrix[i][np.array(adjacency[i])] = 1
            
    else:
        adj_matrix = np.zeros((num_states, num_states))
        for sind, adj_states in adjacency.items():
            ind = int(sind)
            
            if not ind in states:
                raise ValueError('Dictionary index {:d} out of range.'.format(ind))
            
            check_values(adj_states)
            adj_matrix[ind][np.array(adj_states)] = 1
            
    return adj_matrix


