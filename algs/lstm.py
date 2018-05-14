'''
LSTM main module
'''

import numpy as np

from wrapper import Method

from keras.models import Model
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from keras.layers import LSTM, Dense, Input
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import adam

from sklearn.model_selection import GridSearchCV
from keras.layers.core import Masking


def create_model(num_states=1, 
                 max_seq_len=50, 
                 lrate=0.01, 
                 num_units=50, 
                 dropout_rate=0.):
    '''
    Auxiliary function for GridSearchCV
    '''
    in_layer = Input(shape=(max_seq_len, num_states + 1))
#     mask_layer = Masking(mask_value=0)(in_layer)
    lstm_layer = LSTM(units=num_units, dropout=dropout_rate, recurrent_dropout=dropout_rate)(in_layer)
    out_layer = Dense(num_states + 1, activation='softmax', name='output')(lstm_layer)
    
    model = Model(in_layer, out_layer)
    
    opt = adam(lr=lrate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


class MethodLSTM(Method):
    
    def __init__(self, 
                 num_states, 
                 adjacency_list, 
                 max_seq_len, 
                 strat='sequential', 
                 epochs=20, 
                 fname='lstm_model'):
        super(MethodLSTM, self).__init__(num_states, adjacency_list)
        self._max_seq_len = max_seq_len
        self._strat = strat
        self._epochs = epochs
        self._fname = fname
        
    
    def name(self):
        return 'lstm'
    
    def tostr(self):
        return 'lstm_msl{:d}_strat{:s}_epoch{:d}'.format(
            self._max_seq_len,
            'S' if self._strat == 'sequential' else 'G',
            self._epochs)
    
    def validate(self, data):
        # create model
        model = KerasClassifier(build_fn=create_model, 
                                num_states=self._num_states,
                                epochs=self._epochs)
        
        # define the grid search parameters
        batch_size = [32, 64, 96, 128]
        num_units = [50, 75, 100, 125]
        lrate = [0.001, 0.002, 0.005, 0.01]
        dropout_rate = [0.0, 0.1, 0.2]
        
        # full grid-search
        if self._strat == 'grid':
             
            param_grid = dict(max_seq_len=[self._max_seq_len], 
                              lrate=lrate, 
                              num_units=num_units, 
                              dropout_rate=dropout_rate, 
                              batch_size=batch_size)
             
            grid = GridSearchCV(estimator=model, 
                                param_grid=param_grid,
                                cv=10,
                                refit=False, 
                                n_jobs=1, 
                                verbose=0)
            grid_result = grid.fit(data[0], data[1])
              
            grid_result = grid_result.best_params_
         
        # sequential search
        else:         
            order = ['batch_size', 'num_units', 'lrate', 'dropout_rate']
            param_grid = dict(max_seq_len=[self._max_seq_len], 
                              lrate=[lrate[0]], 
                              num_units=[num_units[0]], 
                              dropout_rate=[dropout_rate[0]], 
                              batch_size=[batch_size[0]])
             
            for param in order:
                param_grid[param] = eval(param)
                grid = GridSearchCV(estimator=model, 
                                    param_grid=param_grid,
                                    cv=10,
                                    refit=False, 
                                    n_jobs=1,
                                    verbose=0)
                grid_result = grid.fit(data[0], data[1])
                param_grid[param] = [grid_result.best_params_[param]]
                
            grid_result = {p:v[0]  for p, v in param_grid.items()}
        
        self._best_params = grid_result
    
    
    
    def train(self, data):
        # split data for validation purpose
        idx = np.random.permutation(np.arange(data[0].shape[0]))
        x = data[0][idx]
        y = data[1][idx]
        val_size = int(0.1 * data[0].shape[0])
        x_train, y_train = x[:-val_size], y[:-val_size]
        x_val, y_val = x[-val_size:], y[-val_size:]
        
        self._model = create_model(num_states=self._num_states, 
                                   max_seq_len=self._max_seq_len, 
                                   lrate=self._best_params['lrate'], 
                                   num_units=self._best_params['num_units'], 
                                   dropout_rate=self._best_params['dropout_rate'])
        
        earlystop_cb = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
        modelcheck_cb = ModelCheckpoint(self._fname, monitor='val_loss', save_best_only=True, verbose=0)
        
        hist = self._model.fit(x_train, y_train, 
                               batch_size=self._best_params['batch_size'], 
                               epochs=self._epochs,
                               callbacks=[earlystop_cb, modelcheck_cb], 
                               validation_data=(x_val, y_val),
                               verbose=0)
        
    
    
    def predict(self, sequence, *args):
        # take only the last sample since preparation can make this sequence into multiple samples
        return self._model.predict(sequence[0])[-1][1:], None    
    
    
    def prepare_data(self, data):
        # add additional samples by taking portions of sequences
        ndata = []
        next_state = []
        for i in range(len(data)):
            if len(data[i]) > self._max_seq_len:
                for j in range(0, len(data[i]) - self._max_seq_len):
                    ndata.append(data[i][j: j+self._max_seq_len])
                    next_state.append(data[i][j+self._max_seq_len])
            else:
                ndata.append(data[i][:-1])
                next_state.append(data[i][-1])
            
        ndata = pad_sequences(ndata, self._max_seq_len)  # pad to max length
        
        x = np.zeros((len(ndata), self._max_seq_len, self._num_states + 1), dtype=np.bool)
        y = np.zeros((len(ndata), self._num_states + 1), dtype=np.bool)
        for i, seq in enumerate(ndata):
            for j, state in enumerate(seq):
                x[i, j, state + 1] = 1
            y[i, next_state[i] + 1] = 1
            
        return (x, y)

