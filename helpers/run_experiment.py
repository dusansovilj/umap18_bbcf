'''
'''


import os, pickle
import numpy as np

from helpers import PACKAGE_DIR
from datasets.utils import at_least_trajectories
from helpers.aux import comp_95confint, ensure_directory


RESULTS_DIR = os.path.join(PACKAGE_DIR, 'results')
FIGURES_DIR = os.path.join(PACKAGE_DIR, 'figures')


def run_single_experiment(method, 
                          method_param, 
                          dataset, 
                          dataset_param, 
                          hitratelvl, 
                          fixed_len, 
                          num_folds, 
                          fnsuffix='', 
                          seed=13, 
                          printres=False):
    '''
    Run experiment with given method and dataset.
    Results are automatically saved.
    
    @param method: Method,
        Method to use for the experiment
        
    @param method_param: dict, 
        additional init parameters for Method class
        
    @param dataset: Dataset, 
        Dataset to use for the experiment
        
    @param dataset_param: dict,
        additional init parameters for Dataset class
        
    @param hitratelvl: int, 
        Hit rate to compute in the experiment
        
    @param fixed_len: list(int), 
        Positions in sequences to compute hit rate
        
    @param num_folds: int, 
        How many runs/folds to perform
        
    @param fnsuffix: str, default '',
        file name suffix for saving results 
        
    @param seed: int,
        seed for fold generation
    
    @return: tuple (list(float), list(float)), 
        hit rates for full sequences and
        hit rates for positions at fixed lengths
    '''    
    
    # load data
    data = dataset(**dataset_param)
    data_generator = data.get_data_generator(num_folds, seed)
    
    # make a model
    model = method(
        data.get_num_states(), 
        data.get_adjacency_list(), 
        **method_param)
   
    all_fname = make_filename(model, data, fnsuffix)
    
    if os.path.exists(all_fname):
        hr_all, hr_len = pickle.load(open(all_fname, 'rb'))
        
    else:
        files = []
        
        for idx, d in enumerate(data_generator):
            # check if this fold was already done
            fold_fname = make_filename(model, data, fnsuffix + '-fold{:02}'.format(idx))
            files.append(fold_fname)
            
            if os.path.exists(fold_fname):
                print('skipping fold {:d}'.format(idx))
                continue
            
            # trim test trajectories
            train_fold, test_fold = d
            test_fold = at_least_trajectories(test_fold, fixed_len[-1] + 1)
            
            p_data = model.prepare_data(train_fold)   # prepare data
            model.validate(p_data)                    # validate 
            model.train(p_data)                       # train
            
            hr_all, hr_len = model.compute_hitrate(   # no need to preprocess data here
                test_fold, 
                hitratelvl, 
                fixed_len)  
        
            save_result([hr_all, hr_len], fold_fname)
    

        # collect all results and save
        hr_all = np.zeros((num_folds, hitratelvl))
        hr_len = np.zeros((num_folds, hitratelvl, len(fixed_len)))
        
        for i, f in enumerate(files):
            a, l = pickle.load(open(f, 'rb'))
            hr_all[i] = a
            hr_len[i] = l
            os.remove(f)    # delete file

        pickle.dump([hr_all, hr_len], open(all_fname, 'wb'))

        
    if printres:
        print_results(model.name(), fixed_len, hr_all, hr_len)
    
    return (hr_all, hr_len)



def comp_mean_ci(nparr):
    ''' nparr: 1-dimensional numpy ndarray '''
    if nparr.ndim >= 2:
        nparr = np.reshape(nparr, (-1, ))
    return np.mean(nparr), comp_95confint(nparr)


def comp_mean_ci_4lengths(nparr):
    ''' 
    nparr: 3-dimensional numpy ndarray 
           containing hit rates for specific lengths.
           First dimension are folds. 
    @return ndarray, ndarray: means and confidence intervals
    '''
    assert nparr.ndim == 3
    means = np.zeros(nparr.shape[1:3])
    cis   = np.zeros(nparr.shape[1:3])
    for hr in range(nparr.shape[1]):
        for l in range(nparr.shape[2]):
            m, ci = comp_mean_ci(nparr[:, hr, l])
            means[hr,l] = m
            cis[hr,l]   = ci
    return means, cis
    

def print_results(model_name, fixed_len, hr_all, hr_len):
    '''
    Output the results of the experiment. 
    '''
    N = hr_all.shape[0]
    if hr_all.shape[1] == 1:
        hr_all = np.concatenate([hr_all.reshape(-1,1), np.zeros(N, 1)], axis=1)
        hr_len = np.concatenate([hr_len.reshape(-1, hr_len.shape[1], 1), np.zeros(N, 1, 1)], axis=2)
        
    v = 58
        
    print('\n{:10s}{:8s}{:>10s}{:>10s}{:>10s}{:>10s}\n{:s}'.format('model', 'length', 'hr@1', 'hr@1.ci', 'hr@2', 'hr@2.ci', ''.join(v * ['-'])))
    print('{:s}'.format(''.join(v * ['-'])))
    
    print('{:10s}{:>8s}{:10.2f}{:10.2f}{:10.2f}{:10.2f}'.format(
        model_name, 
        'all', 
        100 * np.mean(hr_all[:,0]), 
        100 * comp_95confint(hr_all[:,0]), 
        100 * np.mean(hr_all[:,1]),  
        100 * comp_95confint(hr_all[:,1])))
    
    for i, l in enumerate(fixed_len):
        print('{:10s}{:>8d}{:10.2f}{:10.2f}{:10.2f}{:10.2f}'.format(
            model_name, 
            l, 
            100 * np.mean(hr_len[:,0,i]), 
            100 * comp_95confint(hr_len[:,0,i]), 
            100 * np.mean(hr_len[:,1,i]), 
            100 * comp_95confint(hr_len[:,1,i])))
    print('{:s}\n'.format(''.join(v * ['-'])))


def make_filename(model, data, extra=''):
    return os.path.join(
        RESULTS_DIR,
        model.name(),
        data.name(),
        '{:s}-{:s}-{:s}.pkl'.format(
            model.tostr(), 
            data.tostr(), 
            extra))


def save_result(lst, fname):
    ensure_directory(fname, True)
    pickle.dump(lst, open(fname, 'wb'))
    
    
def read_result(method, method_param, dataset, dataset_param, fnsuffix=''):
    '''
    Read the results of an experiment. 
    '''
    data = dataset(**dataset_param)
    model = method(
        data.get_num_states(), 
        data.get_adjacency_list(), 
        **method_param)
    
    fname = make_filename(model, data, fnsuffix)
    if os.path.exists(fname):
        hr = pickle.load(open(fname, 'rb'))
        return hr
    else:
        raise ValueError('file not found {:s}'.format(fname))
    
    

    
    
