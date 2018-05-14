'''
Nearest neighbours main module.
In case data contains very long sequences, consider increasing
stride or reducing the number of samples to reduce computational
cost.
'''

import scipy.sparse
from wrapper import Method
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from datasets.utils import convert_to_action_sequence, convert_to_pairwise_sequence
from algs.utils import get_thread_count



class MethodNN(Method):
    
    def __init__(self, 
                 num_states, 
                 adjacency_list, 
                 distfcn='euclidean', 
                 stride=5,
                 algorithm='brute'):
        super(MethodNN, self).__init__(num_states, adjacency_list)
        self._distfcn = distfcn
        self._stride = stride
        self._alg = algorithm
        
        
    def name(self):
        return 'nn'
    
    def tostr(self):
        return 'nn_dst-{:s}_stride{:d}'.format(self._distfcn, self._stride)


    def prepare_data(self, data):
        # add same state before each sequence, so we are
        # able to predict the outcome for single-state samples
        data = [[d[0]] + d  for d in data]
        
        pw_data = None
        pw_actions = []
        
        lens = [len(t)  for t in data]
        for l in range(2, max(lens), self._stride):
            tdata = [ds[:l]  for ds in data if len(ds) > l]
            
            if pw_data is None:
                pw_data = convert_to_pairwise_sequence(tdata, self._num_states)
            else:
                pw_data = scipy.sparse.vstack(
                    (pw_data, 
                     convert_to_pairwise_sequence(tdata, self._num_states)))
            
            pw_actions += [convert_to_action_sequence(seq, self._adjacency_list)[-1]  for seq in tdata]
            
        return pw_data, pw_actions 
    
    
    def validate(self, data):
        param_grid = {'n_neighbors': range(1, 11)}
            
        grid = GridSearchCV(KNeighborsClassifier(
                                metric=self._distfcn, 
                                algorithm=self._alg), 
                            param_grid=param_grid,
                            cv=10,
                            refit=False, 
                            n_jobs=get_thread_count() - 1, 
                            verbose=5)
        
        grid_result = grid.fit(data[0].todense(), data[1])
        
        self._nns = grid_result.best_params_['n_neighbors']


    def train(self, data):
        self._model = KNeighborsClassifier(self._nns, 
                                           metric=self._distfcn, 
                                           algorithm='brute')
        self._model.fit(data[0].todense(), data[1])

        
    def predict(self, sequence, *args):
        return self._model.predict_proba(sequence[0])[-1], None
        
    
