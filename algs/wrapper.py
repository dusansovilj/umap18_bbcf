'''
Wrapper class for algorithms.
'''

import numpy as np


class Method(object):
    
    def __init__(self, num_states, adjacency_list):
        self._num_states = num_states
        self._adjacency_list = adjacency_list
        
    def name(self):
        raise NotImplementedError
    
    
    def tostr(self):
        raise NotImplementedError
    
    
    def train(self, data, **kwargs):
        raise NotImplementedError
    
    
    def validate(self, data):
        return
    
    
    def predict(self, sequence, *args):
        '''
        No batch predictions.
        
        @param sequence: list(list(int)) 
            Sequence to be predicted. First list contains
            only a single element - the sequence itself.
            
        @param *args: type        
            Additional data from previous step.
              
        @return (ndarray, type):
            First output is a numpy array with probabilities
            of transitions towards each other state.
            Second output is auxiliary data for next 
            phase predictions. (default should be None)
        
        '''
        raise NotImplementedError
    
    
    def prepare_data(self, data):
        return data
    
    
    def loglike(self, sequence):
        raise NotImplementedError
    
    
    def next_state(self, sequence, pos):
        '''
        Define the next state in the sequence.
        ''' 
        assert pos >= 0 and pos < len(sequence)
        
        if pos == len(sequence) - 1:
            return None
        else:
            return sequence[pos+1]
    
    
    def compute_hitrate(self, test_data, hrk, lengths, **kwargs):
        '''
        Compute hit rate for test data.
        
        @param test_data: list
            data to compute hr on
        
        @param hrk: int
            hit rate @ k
        
        @param lengths: list(int),
            fixed positions in a sequence
            where to compute hrk  
        
        @return: (ndarray, ndarray):
            hit rate for values up to hrk for complete data; size (hrk,) 
            hit rate for values up to hrk for given lengths; size (hrk, len(lengths)) 
        '''
        
        success_parts = [len(lengths) * [0.]  for _ in range(hrk)]
        success_all = hrk * [0.]
        total_all = 0.
        total_parts = len(lengths) * [0.]
        
        for i in range(len(test_data)):
            prior_trajectory = []
            
            other_args = None  # other arguments for prediction (if needed)
            
            for j in range(len(test_data[i]) - 1):
                prior_trajectory += [test_data[i][j]]
                total_all += 1.

                true_label = self.next_state(test_data[i], j)   # get next state -- prediction target
                
                prepared_data          = self.prepare_data([prior_trajectory + [true_label]])  # add the target value
                prediction, other_args = self.predict(prepared_data, other_args)
                prediction_sorted      = np.argsort(prediction)
                
                # hit rate for the full data
                for x in range(hrk):
                    if len(prediction_sorted) > hrk:
                        if true_label in prediction_sorted[-1-x:]:
                            success_all[x] += 1.
                            
                # hit rate for the specific lengths
                if j + 1 in lengths:
                    idx = next((i for i, x in enumerate(lengths) if x == j + 1), None)
                    total_parts[idx] += 1.
                    for x in range(hrk):
                        if len(prediction_sorted) > hrk:
                            if true_label in prediction_sorted[-1-x:]:
                                success_parts[x][idx] += 1.
                         
        res_all = [s / total_all if total_all > 0 else 0. for s in success_all]
        res_parts = [[x / total_parts[i] if total_parts[i] > 0 else 0. for i, x in enumerate(s)]  for s in success_parts]
        
        res_all = np.array(res_all)
        res_parts = np.array(res_parts)
        
        return res_all, res_parts
    
    
    