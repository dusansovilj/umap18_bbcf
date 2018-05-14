'''
Base abstract class for dataset object.
'''

import os
import cPickle as pickle
from helpers import ensure_directory
from datasets import DATA_DIR


class Dataset(object):
    
    def __init__(self, name):
        self._name = name
        self.make_adjacency_list()
        
        try:
            self.load()
        except IOError:
            self.create_data()
            self.save() 
        
#         fname = self.get_filename()
#         if os.path.exists(fname):
#             self._data = pickle.load(open(fname, 'rb'))
#         else:
#             self.create_data()
#             self.save()
        
    
    def name(self):
        return self._name
    
    def tostr(self):
        raise NotImplementedError
    
    def get_filename(self):
        return os.path.join(DATA_DIR, self.name(), self.tostr() + '.pkl') 
    
    def make_adjacency_list(self):
        ''' 
        Indicate neighboring states for each state.
        Must set self._adjacency_list property. 
        '''
        raise NotImplementedError
    
    def get_adjacency_list(self):
        return self._adjacency_list
    
    def get_num_states(self):
        raise NotImplementedError
    
    def get_data(self):
        raise NotImplementedError
    
    def get_labels(self):
        ''' Return labels for data samples. '''
        raise NotImplementedError
    
    def get_data_generator(self, num_folds, seed):
        ''' make a generator that split data into (train, test) folds '''
        raise NotImplementedError
    
    def create_data(self):
        ''' load/create data; must set self._data property '''
        raise NotImplementedError
    
    def save(self):
        fname = self.get_filename()
        ensure_directory(fname, True)
        pickle.dump(self._data, open(fname, 'wb'))

    def load(self):
        fname = self.get_filename()
        self._data = pickle.load(open(fname, 'rb'))

#     def save(self, fname):
#         fullname = os.path.join(DATA_DIR, 'processed', fname + '.pkl')
#         ensure_directory(fullname, True)
#         pickle.dump(self._data, open(fullname, 'wb'))
#         print('saved')
#        
#     def load(self, fname):
#         fullname = os.path.join(DATA_DIR, 'processed', fname + '.pkl')
#         self._data = pickle.load(open(fullname, 'rb'))
#         print('loaded')
        

