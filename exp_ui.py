'''
Experiments on User-Interface data.
'''

from datasets.userinterface import UIData
from helpers.run_experiment import run_single_experiment 

from algs.markov import MethodMarkov
from algs.bbcf import MethodBBCF
from algs.nearestneigh import MethodNN
from algs.lstm import MethodLSTM


NUM_TEST_SAMPLES = 3
NUM_TASKS = 2

NUM_FOLDS = 10
RND_SEED = 13

FIXED_LENGTHS = [5, 10, 15, 20, 25, 30]
REGULARIZATION_FACTOR = 0.1


fsuffix = 'seed{:d}'.format(RND_SEED)
show_result = True

def main():
    run_single_experiment(
        MethodMarkov, 
        {'reg_factor': REGULARIZATION_FACTOR}, 
        UIData, 
        {'num_test': NUM_TEST_SAMPLES},
        2, 
        FIXED_LENGTHS,
        10,
        fsuffix,
        RND_SEED,
        show_result)
        
        
    run_single_experiment(
        MethodBBCF, 
        {'reg_factor': REGULARIZATION_FACTOR, 'num_models': NUM_TASKS}, 
        UIData, 
        {'num_test': NUM_TEST_SAMPLES},
        2, 
        FIXED_LENGTHS,
        10,
        fsuffix,
        RND_SEED,
        show_result)
    
     
    run_single_experiment(
        MethodNN, 
        {'distfcn': 'euclidean', 'stride': 1},
        UIData, 
        {'num_test': NUM_TEST_SAMPLES},
        2, 
        FIXED_LENGTHS, 
        10, 
        fsuffix,
        RND_SEED,
        show_result)
     
      
    run_single_experiment(
        MethodNN, 
        {'distfcn': 'euclidean', 'stride': 1},
        UIData, 
        {'num_test': NUM_TEST_SAMPLES},
        2, 
        FIXED_LENGTHS, 
        10, 
        fsuffix,
        RND_SEED,
        show_result)
     
     
    run_single_experiment(
        MethodLSTM, 
        {'max_seq_len': 30, 'strat': 'sequential', 'epochs': 20, 'fname': 'lstm_model'}, 
        UIData, 
        {'num_test': NUM_TEST_SAMPLES},
        2, 
        FIXED_LENGTHS, 
        10, 
        fsuffix,
        RND_SEED,
        show_result)


if __name__ == '__main__':
    main()
    
    