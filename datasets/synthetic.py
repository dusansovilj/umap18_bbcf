'''
Synthetic data set for task separability experiment
'''

import numpy as np
from datasets.wrapper import Dataset
from sklearn.model_selection import KFold


class SyntheticData(Dataset):
    
    num_states = 4
    
    # transition matrices
    Tmat = [np.asarray([[0.25, 0.75, 0., 0.], [0.1, 0.1, 0.8, 0.], [0.9, 0.1, 0., 0.], [0., 0., 0., 1.]]), 
            np.asarray([[0.2, 0., 0., 0.8], [0., 0.1, 0.9, 0.], [0., 0.7, 0.3, 0.], [0.95, 0., 0., 0.05]]),
            np.asarray([[0., 0., 1., 0.], [0., 0., 0., 1.], [0., 1., 0., 0.], [1., 0., 0., 0.]])]
    
    def __init__(self, num_samples=5000, seplvl=0.5, seed=1):
        self._num_samples = num_samples
        self._seplvl = seplvl
        self._seed = seed
        super(SyntheticData, self).__init__('synthetic')
        
        
    def tostr(self):
        return '{:s}_ns{:d}_sl{:.2f}_seed{:d}'.format(
            self.name(),
            self._num_samples,
            self._seplvl,
            self._seed)

        
    def create_data(self):
        num_states = SyntheticData.num_states
        Tuni = 1. / num_states * np.ones((num_states, num_states))
        
        Tnew = [self._seplvl * T + (1 - self._seplvl) * Tuni  for T in SyntheticData.Tmat]
        
        rs = np.random.RandomState(self._seed)
        
        self._data = []
        for task in range(0, len(SyntheticData.Tmat)):
            for _ in range(self._num_samples):
                inis = rs.permutation(num_states)[0]
                ln = rs.randint(low=5, high=50)
                
                datum = [inis]
                for _ in range(ln):
                    new_state = np.argmax(rs.multinomial(1, Tnew[task][datum[-1]]))
                    datum.append(new_state)
                    
                datum.append(num_states + task)  # append goal/task state
                self._data.append(datum)
                
        
    def get_num_states(self):
        return SyntheticData.num_states + len(SyntheticData.Tmat)
    
    
    def make_adjacency_list(self):
        res = [[]] * (SyntheticData.num_states + len(SyntheticData.Tmat))
        task_states = range(
            SyntheticData.num_states, 
            SyntheticData.num_states + len(SyntheticData.Tmat))
        
        all_states = range(SyntheticData.num_states) + task_states
        for i in range(SyntheticData.num_states):
            res[i] = all_states
        for i in task_states:
            res[i] = [i]
            
        self._adjacency_list = res
    
    
    def get_data(self):
        return self._data

    
    def get_labels(self):
        return [s[-1] - SyntheticData.num_states  for s in self._data]
    
    
    def get_data_generator(self, num_folds=10, seed=1):
        kf = KFold(n_splits=num_folds, random_state=seed)
        kf.get_n_splits(range(len(self._data)))
        
        for train_idx, test_idx in kf.split(self._data):
            train = [self._data[idx]  for idx in train_idx]      
            test  = [self._data[idx]  for idx in test_idx]
            yield train, test
        
        