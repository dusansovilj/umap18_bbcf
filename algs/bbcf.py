'''
BBCF main module  
'''


import numpy as np
from scipy.sparse import coo_matrix

from math import exp,log

from algs.utils import get_next_state_probability
from algs.utils import get_goal_state_distribution

from algs.wrapper import Method


class MethodBBCF(Method):
    
    def __init__(self, num_states, adjacency_list, reg_factor, num_models):
        super(MethodBBCF, self).__init__(num_states, adjacency_list)
        self._rf = reg_factor
        self._num_models = num_models   # number of models to use (limit to most dominant final states) 
    
    
    def name(self):
        return 'bbcf'
    
    def tostr(self):
        return 'bbcf_rf{:.2f}_nm{:d}'.format(self._rf, self._num_models)
    
        
    def train(self, data):
        self._goal_state_label = get_goal_state_label(self._num_models, 
                                                      self._num_states, 
                                                      data)
        
        self._T = get_goal_transition_matrices(data, 
                                               self._num_states, 
                                               self._goal_state_label, 
                                               self._adjacency_list, 
                                               self._rf)
        
        self._initial_state_prob = get_initial_state_probability(data, 
                                                                 self._num_states, 
                                                                 self._rf)
        
        self._goal_prior_prob = P_T_frequency(data,
                                              self._num_states)
        
    
    def predict(self, sequence, *args):
#         res, _, _ = self.P_a_c(sequence[0][:-1])
#         return res
        res, loglikes = self.P_a_c(sequence[0][:-1], args[0])  # expect log-likelihoods here
        return res, loglikes
        
        
    # P(a | C)
#     def P_a_c(self, trajectory):
#         '''
#         Return the next_state probability given the prior_trajectory
#         Input: training data, number_of_states(400 or 14),number_of_task(400 or 4)
#                epsilon(regularization parameter),
#                T_res(contains all goal-transition matrix)
#                Initial_state_probability(P(C_0))
#                frequency(P(T_g))
#         '''
#         res = np.zeros(self._num_states)
#         
#         P_C, likelihood_record = self.p_c(trajectory)      # record the likelihood of the trajectory
#         
#         # Sum over all goal-states
#         weight_record = []
#         for goal in self._goal_state_label:
#             x = self.P_a_TC(goal, trajectory)              # P(a|T_g,C) probability of next_state
#             y = self.P_T_C(goal, likelihood_record, P_C)   # P(T|C) weights for different goal-states
#             res += x * y 
#             weight_record.append(y)
#             
#         loglikes = [log(likes) if likes > 0 else log(self._rf) for likes in likelihood_record]
#         return res, weight_record, loglikes
    
    # P(a | c) - probability of an action given a trajectory
    def P_a_c(self, trajectory, loglikes=None):
        res = np.zeros(self._num_states)
        P_C, loglikes = self.p_c(trajectory, loglikes)
        
        for goal in self._goal_state_label:
            x = self.P_a_TC(goal, trajectory)
            y = self.P_T_C_v2(goal, loglikes, P_C)
            res += x * y
        return res, loglikes
    
    
#     # P(c) - probability of a trajectory
#     def p_c(self, trajectory):
#         ''' Calculate the P(C) and record the likelihood for other computation '''
#         res = 0
#         likelihood_record = np.zeros(self._num_states)
#         
#         for goal in self._goal_state_label:
#             log_likelihood = self.log_P_C_T(goal, trajectory)
#             likelihood = exp(log_likelihood)
#             likelihood_record[goal] += (likelihood)
#             res += likelihood * self._goal_prior_prob[goal]
#             
#         return res, likelihood_record
    
    
    # P(c) - probability of a trajectory
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
    
    
#     def log_P_C_T(self, goal_state, trajectory):
#         '''Get the log likelihood of the prior trajectory'''
#         log_sum = 0
#         T = self._T[goal_state]
#         
#         # P(C_i | C_{i-1}, T_g)
#         for i in range(len(trajectory) - 1):
#             previous_state = trajectory[i]
#             action_probability = get_next_state_probability(T, previous_state)  #get probability
#             if action_probability[trajectory[i+1]] > 0:
#                 log_sum += log(action_probability[trajectory[i+1]])
#             else:
#                 log_sum += log(self._rf)
#                 
#         # P(C_0)
#         p_c_one = self._initial_state_prob[goal_state][trajectory[0]]
#         log_p_c_one = log(p_c_one)
#         log_sum += log_p_c_one
#         return log_sum
    
    
    def log_P_C_T(self, goal_state, trajectory, loglike=None):
        '''
        Return log-likelihood of a trajectory given transition model. Add the last state
        of trajectory into log-like computation given specific goal state.
        If log-likelihood is not provided, it is computed from scratch. 
        '''
        if loglike is None:
            # P(C_0)
            p_c_one = self._initial_state_prob[goal_state][trajectory[0]]
            log_p_c_one = log(p_c_one)
            log_sum = log_p_c_one
            start_pos = 0
        else:
            log_sum = loglike
            start_pos = len(trajectory) - 2
    
        T = self._T[goal_state]
        
        # P(C_i | C_{i-1}, T_g)
        for i in range(start_pos, len(trajectory) - 1):
            previous_state = trajectory[i]
            action_probability = get_next_state_probability(T, previous_state)
            log_sum += log(action_probability[trajectory[i+1]])  if action_probability[trajectory[i+1]] > 0 else log(self._rf)
            
        return log_sum
        
    
    
    # P(a | T_g, C)
    def P_a_TC(self, goal_state, trajectory):
        '''
        P(a|T_g,C) 
        Get the next_state probability distribution based on the Goal_transition matrix 
        and the last element from the prior-trajectory
        '''
        T = self._T[goal_state]                                # Get goal-transition matrix from the collection of transition matrix
        action = get_next_state_probability(T, trajectory[-1]) # Get next_state prob. dist. given T and last element from prior_trajectory
        return action
    
    
    # P(T_g | C)
    def P_T_C(self, goal_state, likelihood_record, P_C):
        '''
        P(T|C) input goal_state and the likelihood_record collected by P(C)
        Output the weights of next_action probability
        '''
        P_C_T = likelihood_record[goal_state]
        P_T = self._goal_prior_prob[goal_state]
        return P_C_T * P_T / P_C

    def P_T_C_v2(self, goal_state, log_likelihood, P_C):
        P_C_T = exp(log_likelihood[goal_state])
        P_T = self._goal_prior_prob[goal_state]
        return P_C_T * P_T / P_C
    

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
            
#     for idx in range(len(goal_state_label)):
#         row_sums = res[idx].sum(axis=1)
#         res[idx] = res[idx] / row_sums[:, np.newaxis]    
        
    return T


def get_goal_state_label(first_k_goal, num_of_state, data):
    ''' Get the goal states label that are going to contribute to collaborate filtering algorithms '''
    goal_dict = get_goal_state_distribution(data)
    num_of_state = min([num_of_state, len(goal_dict)])
    ind = list(np.argsort(goal_dict.values()))
    return np.array(goal_dict.keys())[ind[(num_of_state - first_k_goal):]]


def get_initial_state_probability(data, num_states, epsilon):
    '''Go over the training set and count the frequency of the start_state probability distribution'''
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


    
    
    