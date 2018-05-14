'''
'''

import numpy as np
from scipy.sparse import lil_matrix
import sys
from sklearn.model_selection import train_test_split


def get_train_and_test_set(output_data, percent, seed=0):
    '''
    Input the data of trajectories and percentage of training data you want, 
    return train and test data at the percentage of (for example) 80% : 20% randomly
    '''
# #     new_data = np.random.shuffle(output_data)
#     new_data = output_data
#     temp = int(len(output_data) * percent)
#     train_data = new_data[:temp]
#     test_data = new_data[temp:]
#     return train_data, test_data

    train_data, test_data = train_test_split(output_data, train_size=int(len(output_data) * percent), random_state=seed)
    return train_data, test_data


def trim_trajectories(data, length):
    if length <= 0:
        print('Length should be positive value (integer).')
        sys.exit()
        
    length = int(length)
    for i in range(len(data)):
        data[i] = data[i][:length]
    return data
    
    
def remove_selftransitions(data):
    '''Edit the data such that there is no self-transition state exists'''
    new_data = []
    for i in range(len(data)):
        temp = [data[i][0]]
        for j in range(len(data[i])):
            if(data[i][j]!=temp[-1]):
                temp.append(data[i][j])
        new_data.append(temp)
    return new_data


def fixed_len_trajectories(data, length):
    '''
    Return only trajectories that are at least length long, and cut them at that point
    '''
    i = 0
    while i < len(data):
        if len(data[i]) >= length:
            data[i] = data[i][:length]
            i += 1
        else:
            del data[i]
    return data


def at_least_trajectories(data, length):
    '''
    Return trajectories that are at least length long
    '''
    i = 0
    while i < len(data):
        if len(data[i]) < length:
            del data[i]
        else:
            i += 1
    return data


def convert_to_pairwise_sequence(state_seq, num_states):
    '''
    Convert state sequence to a pairwise state sequence.
    '''
    N = len(state_seq)
    new_seq = lil_matrix((N, num_states * num_states), dtype='int16')
    for i in range(N):
        for k in range(1, len(state_seq[i])):
            new_seq[i, state_seq[i][k-1] * num_states + state_seq[i][k]] += 1
        
    return new_seq


def convert_to_action_sequence(state_seq, adjacency_list):
    '''
    Convert state sequence to action sequence with predefined adjacency list
    '''
    
    act_seq = []
    for i in range(1, len(state_seq)):
#         current_state = state_seq[i]
#         next_state = state_seq[i-1]
#         act_idx = next((i for i, x in enumerate(adjacency_list[current_state]) if x == next_state), None)
        act_idx = action_performed(state_seq, i-1, adjacency_list)
        act_seq.append(act_idx)
    return act_seq


def create_action_matrices(data, num_actions, adjacency_list):
    '''
    Convert state sequences to action sequences, where an action
    is some indicator transferring state[i] to state[i+1].
    
    For taxi data, an action set is  {stay, top-left, left, etc.} with 9 total actions.
    For UI data, there are predefined actoins {mouse zoom, rcl scroll, etc.} with total of 14 actions.
    '''
    
    N = len(data)
    new_seq = [convert_to_action_sequence(s, adjacency_list)  for s in data]
    
    new_train = np.zeros((len(data), num_actions * num_actions), dtype='int16')
    for i in range(N):
        for j in range(1, len(new_seq[i])): 
            new_train[new_seq[i][j-1]][new_seq[i][j]] += 1
    
    new_train = lil_matrix(new_train)
    return new_train


def action_performed(seq, pos, adjacency_list):
    '''
    Get a numbered action for a position in a sequence 
    given the indexing in adjacency list. 
    '''
    if pos < 0 and pos >= len(seq) - 1:
        return None
    
    current_state = seq[pos]
    next_state = seq[pos+1]
    return next((i for i, x in enumerate(adjacency_list[current_state]) if x == next_state), None)



def get_length_of_trajectories(data):
    ''' Get the length distribution of trajectories '''
    return [len(e)  for e in data]


def is_consistent(trajectory, adjacency_list):
    ''' Check if list of states adheres to its adjacency structure '''
    for i in range(1, len(trajectory)):
        if not trajectory[i] in adjacency_list[trajectory[i-1]]:
            return False

    return True


def get_total_length_record(data):
    length_total_record = []
    for i in range(10):
        temp = 0
        for j in range(10-i):
            temp += get_length_of_trajectories(data)[j+i]
        length_total_record.append(temp)
    return length_total_record

