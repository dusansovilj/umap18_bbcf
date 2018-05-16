'''
Experiments on Taxi data for variable number of samples.
'''

from helpers.run_experiment import run_single_experiment
from datasets.taxisparse import SparseTaxiData

from algs.markov import MethodMarkov
from algs.bbcf import MethodBBCF
from algs.nearestneigh import MethodNN
from algs.lstm import MethodLSTM

LOAD_SIZE = 100000   # how many rows to load from taxi train file
TRAIN_SIZE = 1000     # choose 200, 400, 600, 800, 1000
TEST_SIZE = 500      # fix the number of test samples

GRID_SIZE = [20, 20]
NUM_STATES = GRID_SIZE[0] * GRID_SIZE[1]
REGULARIZATION_FACTOR = 0.1
NUM_GOALS = 10       # how many goal states to take into account
NUM_FOLDS = 10
RND_SEED = 11

FIXED_LENGTHS = [2, 3, 4, 5, 6, 7, 8, 9, 10]


fsuffix = 'seed{:d}'.format(RND_SEED)
show_result = True


def main():
    print('{:s}\nNumber of training samples {:d}\n{:s}'.format(''.join(40 * ['-']), TRAIN_SIZE, ''.join(40 * ['-'])))
    
    run_single_experiment(
        MethodMarkov,
        {'reg_factor': REGULARIZATION_FACTOR},
        SparseTaxiData, 
        {'num_samples': LOAD_SIZE, 'num_train': TRAIN_SIZE, 'num_test': TEST_SIZE, 'num_goals': NUM_GOALS, 'grid': [20, 20]},
        2, 
        FIXED_LENGTHS,
        10,
        fsuffix,
        RND_SEED,
        show_result)
    
    # number of destinations is limited  
    run_single_experiment(
        MethodBBCF,
        {'reg_factor': REGULARIZATION_FACTOR, 'num_models': NUM_GOALS},
        SparseTaxiData, 
        {'num_samples': LOAD_SIZE, 'num_train': TRAIN_SIZE, 'num_test': TEST_SIZE, 'num_goals': NUM_GOALS, 'grid': [20, 20]},
        2, 
        FIXED_LENGTHS,
        10,
        fsuffix,
        RND_SEED,
        show_result)
    
    
    run_single_experiment(
        MethodNN, 
        {'distfcn': 'euclidean', 'stride': 3},
        SparseTaxiData, 
        {'num_samples': LOAD_SIZE, 'num_train': TRAIN_SIZE, 'num_test': TEST_SIZE, 'num_goals': NUM_GOALS, 'grid': [20, 20]}, 
        2, 
        FIXED_LENGTHS, 
        10, 
        fsuffix,
        RND_SEED,
        show_result)
    
     
    run_single_experiment(
        MethodNN, 
        {'distfcn': 'cosine', 'stride': 3},
        SparseTaxiData, 
        {'num_samples': LOAD_SIZE, 'num_train': TRAIN_SIZE, 'num_test': TEST_SIZE, 'num_goals': NUM_GOALS, 'grid': [20, 20]},
        2, 
        FIXED_LENGTHS, 
        10, 
        fsuffix,
        RND_SEED)
    
    
    run_single_experiment(
        MethodLSTM, 
        {'max_seq_len': 10, 'strat': 'sequential', 'epochs': 20, 'fname': 'lstm_model'}, 
        SparseTaxiData, 
        {'num_samples': LOAD_SIZE, 'num_train': TRAIN_SIZE, 'num_test': TEST_SIZE, 'num_goals': NUM_GOALS, 'grid': [20, 20]},
        2, 
        FIXED_LENGTHS, 
        10, 
        fsuffix,
        RND_SEED,
        show_result)


if __name__ == '__main__':
    main()
    
    