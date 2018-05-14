'''
'''

import cPickle as pickle
import os
import itertools
from sklearn.model_selection import KFold
from datasets.wrapper import Dataset
from datasets import DATA_DIR

PREPROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'kt', 'raw', 'ast_skmult.pkl')


class KTData(Dataset):
    
    NUM_STATES = 2
    
    def __init__(self, skill_set=0, merge=True):
        self._skill_set = skill_set
        self._merge = merge
        super(KTData, self).__init__('kt')
        
    def tostr(self):
        return '{:s}_skillset{:d}'.format(self.name(), self._skill_set)
    
    def save(self):
        super(KTData, self).save(self.tostr())
        
    def load(self):
        super(KTData, self).load(self.tostr())

    def make_adjacency_list(self):
        self._adjacency_list = [[0, 1], [0, 1]]
    
    def get_num_states(self):
        return KTData.NUM_STATES
    
    def get_data(self):
        return self._data
    
    def create_data(self):
        self._data = load_kt_data(self._skill_set, self._merge)
    
    def get_labels(self):
        return [self._skill_set] * len(self._data)
    
    def get_data_generator(self, num_folds, seed):
        kf = KFold(n_splits=num_folds, random_state=seed)
        kf.get_n_splits(range(len(self._data)))
        
        for train_idx, test_idx in kf.split(self._data):
            train = [self._data[idx]  for idx in train_idx]      
            test  = [self._data[idx]  for idx in test_idx]
            yield train, test



def load_kt_data(skill_set=0, merge=True):
    f = open(PREPROCESSED_DATA_FILE, 'rb')
    data = pickle.load(f)
    f.close()
    
    data = data[skill_set]
    if merge:
        data = [list(itertools.chain(*x))  for x in data]
        data = [map(int, sample) for sample in data]
    else:
        for i, user in enumerate(data):
            for j, _ in enumerate(user):
                data[i][j] = map(int, data[i][j])

    return data


