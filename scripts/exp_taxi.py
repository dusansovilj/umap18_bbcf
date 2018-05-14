'''
Taxi data experiments.
'''

from datasets.taxi import TaxiData

from run_experiment import run_single_experiment

from algs.markov import MethodMarkov
from algs.bbcf import MethodBBCF
from algs.nearestneigh import MethodNN
from algs.lstm import MethodLSTM



FIXED_LENGTHS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
TRAIN_SIZE = 100000
NUM_GOALS = 200              # how many goal states to take into account

GRID_SIZE = [20, 20]
NUM_STATES = GRID_SIZE[0] * GRID_SIZE[1]
REGULARIZATION_FACTOR = 0.1
NUM_FOLDS = 10
RND_SEED = 13


fsuffix = 'seed{:d}'.format(RND_SEED)
show_result = True


def main():
    run_single_experiment(
        MethodMarkov, 
        {'reg_factor': REGULARIZATION_FACTOR}, 
        TaxiData, 
        {'num_samples': TRAIN_SIZE, 'grid': GRID_SIZE},
        2, 
        FIXED_LENGTHS,
        10,
        fsuffix,
        RND_SEED,
        show_result)
     
     
    run_single_experiment(
        MethodBBCF,
        {'reg_factor': REGULARIZATION_FACTOR, 'num_models': NUM_GOALS},
        TaxiData,
        {'num_samples': TRAIN_SIZE, 'grid': GRID_SIZE},
        2, 
        FIXED_LENGTHS, 
        10, 
        fsuffix,
        RND_SEED,
        show_result)
      
      
    run_single_experiment(
        MethodNN, 
        {'distfcn': 'euclidean', 'stride': 5},
        TaxiData, 
        {'num_samples': TRAIN_SIZE, 'grid': GRID_SIZE},
        2, 
        FIXED_LENGTHS, 
        10, 
        fsuffix,
        RND_SEED,
        show_result)
      
       
    run_single_experiment(
        MethodNN, 
        {'distfcn': 'cosine', 'stride': 5},
        TaxiData, 
        {'num_samples': TRAIN_SIZE, 'grid': GRID_SIZE},
        2, 
        FIXED_LENGTHS, 
        10, 
        fsuffix,
        RND_SEED,
        show_result)
      
      
    run_single_experiment(
        MethodLSTM, 
        {'max_seq_len': 10, 'strat': 'sequential', 'epochs': 20, 'fname': 'lstm_model'}, 
        TaxiData, 
        {'num_samples': TRAIN_SIZE, 'grid': GRID_SIZE},
        2, 
        FIXED_LENGTHS, 
        10, 
        fsuffix,
        RND_SEED,
        show_result)


if __name__ == '__main__':
    main()
    
    