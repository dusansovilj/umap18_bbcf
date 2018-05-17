'''
BBCF main module  
'''


import numpy as np
from scipy.sparse import coo_matrix

from math import exp

from algs.utils import get_goal_state_distribution
from algs.wrapper import Method
from algs.markov import MethodMarkov


class MethodBBCF(Method):
    
    def __init__(self, num_states, adjacency_list, reg_factor, num_models):
        super(MethodBBCF, self).__init__(num_states, adjacency_list)
        self._rf = reg_factor
        self._num_models = num_models   # number of models to use (limit to most dominant final states)
        self._models = {}   # list of Method objects 
    
    
    def name(self):
        return 'bbcf'
    
    def tostr(self):
        return 'bbcf_rf{:.2f}_nm{:d}'.format(self._rf, self._num_models)
    
        
    def train(self, data):
        self._goal_state_label = get_goal_state_label(self._num_models, 
                                                      self._num_states, 
                                                      data)
        
        # modify the model initialization to accommodate 
        # different learning algorithms
        # initialize models
        for goal in self._goal_state_label:
            temp_model = MethodMarkov(
                self._num_states, 
                self._adjacency_list, 
                self._rf)
            temp_data = [s  for s in data if s[-1] == goal]
#             temp_data = temp_model.prepare_data(temp_data)
            temp_model.train(temp_data)
            self._models[goal] = temp_model
        
        self._goal_prior_prob = P_T_frequency(data,
                                              self._num_states)
        
    
    def predict(self, sequence, *args):
#         res, _, _ = self.P_a_c(sequence[0][:-1])
#         return res
        res, loglikes = self.P_a_c(sequence[0][:-1], args[0])  # expect log-likelihoods here
        return res, loglikes
        
    
    # P(a | c) - probability of an action given a trajectory
    def P_a_c(self, trajectory, loglikes=None):
        res = np.zeros(self._num_states)
        P_C, loglikes = self.p_c(trajectory, loglikes)
        
        for goal in self._goal_state_label:
            x = self.P_a_TC(goal, trajectory)
            y = exp(loglikes[goal]) * self._goal_prior_prob[goal] / P_C  # P(C|T) * P(T) / P(C) = P(T|C) - posterior of a model
            res += x * y
        return res, loglikes
    
    
    # P(c) - probability of a trajectory
    # P(c) = sum_{t=1}^T P(t) * P(c|t),  
    #        prior goal model probability * likelihood under that goal model
    def p_c(self, trajectory, loglikes=None):
        res = 0
        if loglikes is None:
            loglikes = np.zeros(self._num_states)
            for goal in self._goal_state_label:
                loglikes[goal] += self.log_P_C_T(goal, trajectory)
                res += exp(loglikes[goal]) * self._goal_prior_prob[goal]
                
        else:
            for goal in self._goal_state_label:
                loglikes[goal] = self.log_P_C_T(goal, trajectory, loglikes[goal])
                res += exp(loglikes[goal]) * self._goal_prior_prob[goal]
                
        return res, loglikes
    
    
    def loglike(self, sequence, *args):
        full_loglike, _ = self.p_c(sequence, args[0])
        return full_loglike 

    
    def log_P_C_T(self, goal_state, trajectory, loglike=None):
        '''
        Return log-likelihood of a trajectory given transition model. 
        If log-likelihood is not provided, it should be computed from scratch. 
        '''
        return self._models[goal_state].loglike(trajectory, loglike)
        
    
    # P(a | T_g, C)
    def P_a_TC(self, goal_state, trajectory):
        '''
        Get the next_state probability distribution based on 
        the destination state and trajectory.
        '''
        temp_trajectory = [trajectory + [None]]
        temp_data = self._models[goal_state].prepare_data(temp_trajectory)
        action, _ = self._models[goal_state].predict(temp_data)
        return action
    
    

def get_specific_goal_transition(train_data, num_of_state, goal_state_label, adjacent_state, epsilon):
    '''Get T_g with selected goal-states rather than all goal states. This is for optimizing the calculation efficiency
    since not all goal states contribute much to the trajectory prediction.
    '''
    res = []
    for i in range(num_of_state):
        if(i in goal_state_label): # Calculate the specific goal states that were marked
            T = T_g(i,train_data,num_of_state,adjacent_state,epsilon)
            res.append(T)
        else: # Otherwise fill out the blank with empty list
            res.append([])
    return res



##################################
#####    aux functions   #########
##################################

def T_g(goal_state, train_data, num_of_state, adjacent_state, epsilon):
    '''Get transition matrix with specific goal state'''
    row = []
    col = []
    value = []
    
    # Counting 
    for i in range(len(train_data)):
        if (goal_state == train_data[i][-1]):
            for j in range(len(train_data[i])-1):
                current_state = train_data[i][j]
                next_state = train_data[i][j+1]
                row.append(current_state)
                col.append(next_state)
                value.append(1)
                
    # Regularization
    for i in range(len(adjacent_state)):
        for j in range(len(adjacent_state[i])):
            if(adjacent_state[i][j] == 1):
                row.append(i)
                col.append(j)
                value.append(epsilon)
                
    T = coo_matrix((value, (row, col)), shape=(num_of_state, num_of_state)).toarray()
    
    # Normalization
    row_sums = T.sum(axis=1)
    new_matrix = T / row_sums[:, np.newaxis]
    
    return new_matrix


def get_goal_transition_matrices(train_data, num_of_state, goal_state_label, adjacency_list, epsilon):
#     res = epsilon * np.ones((len(goal_state_label), num_of_state, num_of_state))
    res = np.zeros((len(goal_state_label), num_of_state, num_of_state))
    
    for i in range(len(train_data)):
        goal_state = train_data[i][-1]
        if not goal_state in goal_state_label:
            continue
        
        idx = np.nonzero(goal_state == goal_state_label)
        for j in range(len(train_data[i]) - 1):
            current_state = train_data[i][j]
            next_state    = train_data[i][j+1]
            res[idx, current_state, next_state] += 1
            
    for i in range(len(adjacency_list)):
        for s in adjacency_list[i]:
            res[:, i, s] += epsilon
        
    # Normalization
    T = []
    for i in range(num_of_state):
        idx = np.nonzero(i == goal_state_label)
        if len(idx) > 0:
            m = np.squeeze(res[idx])
            row_sums = m.sum(axis=1)
            m = m / row_sums[:, np.newaxis]
            T.append(m)
        else:
            T.append([])
            
    return T


def get_goal_state_label(first_k_goal, num_of_state, data):
    ''' Get the goal states label that are going to contribute to collaborate filtering algorithms '''
    goal_dict = get_goal_state_distribution(data)
    num_of_state = min([num_of_state, len(goal_dict)])
    ind = list(np.argsort(goal_dict.values()))
    return np.array(goal_dict.keys())[ind[(num_of_state - first_k_goal):]]


def get_initial_state_probability(data, num_states, epsilon):
    '''
    Go over the training set and count the frequency of the start_state 
    probability distribution for each destination '''
    distribution = np.zeros((num_states, num_states))
    for i in range(num_states):
        for j in range(len(data)):
            if(data[j][-1] == i):
                distribution[i][data[j][0]] += 1
                
        for x in range(num_states):
            distribution[i][x] += epsilon

    row_sums = distribution.sum(axis=1)
    new_matrix = distribution / row_sums[:, np.newaxis]
    return new_matrix

    


def P_T_frequency(train_data, max_goals):
    '''
    Prior distribution over goal states P(T_g) based on frequency
    '''
    frequency = np.zeros(max_goals, dtype='float32')
    for i in range(len(train_data)):
        end_state = train_data[i][-1]
        frequency[end_state] += 1.
    return frequency / sum(frequency)


    
    
    