'''
Taxi data for sparsity test.
The data is the same as in full TaxiData, but 
naming convention makes additional pickle files
in data directory.
'''

import numpy as np
from datasets.taxi import TaxiData


class SparseTaxiData(TaxiData):
    
    def __init__(self, num_samples, num_train, num_test, num_goals, grid=[20, 20]):
        self._n_train = num_train
        self._n_test = num_test
        self._n_goals = num_goals        # max number of destination states
        super(SparseTaxiData, self).__init__(num_samples, grid)
        
    def create_data(self):
        super(SparseTaxiData, self).create_data()
        
    def name(self):
        return 'taxisparse'
    
    def tostr(self):
        return  '{:s}_ns{:d}_train{:d}_test{:d}_grid{:d}-{:d}'.format(
            self.name(),
            self._num_samples,
            self._n_train,
            self._n_test,
            self._grid[0],
            self._grid[1])
    
    def make_adjacency_list(self):
        super(SparseTaxiData, self).make_adjacency_list()
    
    
    def get_data_generator(self, num_folds, seed):
        rs = np.random.RandomState(seed)
        idx = rs.permutation(range(len(self._data)))
        
        data = [self._data[i]  for i in idx]
        data_test = data[-self._n_test:]    # fix the test data
        data = data[:-self._n_test]
        
        # ============================================
        # focus only on samples that are among the top 
        # frequent goals
        # ============================================
            
        gs = np.array([p[-1]  for p in data], dtype='int32')
        top_goal_states = list(np.argsort(np.bincount(gs))[-self._n_goals:])
        data = [d  for d in data if d[-1] in top_goal_states]
        
        def get_trajectories_with_goal(seqs, goal):
            return [s  for s in seqs if s[-1] == goal]
            
        def split_into_equal_folds(data, num_folds):
            foldsize = len(data) / num_folds
            folds = []
            for k in range(num_folds):
                idx = range(k * foldsize, (k+1) * foldsize)
                folds.append([data[j] for j in idx])
            return folds
        
        dd = dict()
        for g in top_goal_states:
            gdata  = get_trajectories_with_goal(data, g)
            gfolds = split_into_equal_folds(gdata, num_folds)
            dd[g]  = gfolds
            
            
        foldsize = len(data) / num_folds
        fold_ids = []
        for k in range(num_folds):
            idx = range(k * foldsize, (k+1) * foldsize)
            fold_ids.append(idx)
            
        samples_per_task = self._n_train / self._n_goals
            
        for c in range(1, num_folds+1):
            data_fold = []
            for g in dd.keys():
                data_fold += dd[g][c - 1][:samples_per_task]
                
            yield data_fold, data_test
                
                
                