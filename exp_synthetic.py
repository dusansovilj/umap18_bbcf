'''
Synthetic data experiments
'''


from datasets.synthetic import SyntheticData

from helpers.run_experiment import run_single_experiment
from algs.markov import MethodMarkov
from algs.bbcf import MethodBBCF
from algs.nearestneigh import MethodNN
from algs.lstm import MethodLSTM


TASK_SEP = 1      # change this for varying degrees of separability -- [0., 0.2, 0.4, 0.6, 0.8, 1.]

FIXED_LENGTHS = [5, 6, 7, 8, 9, 10, 11, 12]
REGULARIZATION_FACTOR = 0.1
NUM_TRAINING_SAMPLES = 5000
RND_SEED = 13
NUM_FOLDS = 10
NUM_TASKS = 3

fsuffix = 'seed{:d}'.format(RND_SEED)
show_result = True


def main():
    run_single_experiment(
        MethodMarkov, 
        {'reg_factor': REGULARIZATION_FACTOR}, 
        SyntheticData, 
        {'num_samples': NUM_TRAINING_SAMPLES, 'seplvl': TASK_SEP, 'seed': RND_SEED},
        2, 
        FIXED_LENGTHS,
        NUM_FOLDS,
        fsuffix,
        RND_SEED,
        show_result)


    run_single_experiment(
        MethodBBCF, 
        {'reg_factor': REGULARIZATION_FACTOR, 'num_models': NUM_TASKS},
        SyntheticData, 
        {'num_samples': NUM_TRAINING_SAMPLES, 'seplvl': TASK_SEP, 'seed': RND_SEED},
        2, 
        FIXED_LENGTHS, 
        10, 
        fsuffix,
        RND_SEED,
        show_result)
       
        
    run_single_experiment(
        MethodNN, 
        {'distfcn': 'euclidean', 'stride': 3},
        SyntheticData, 
        {'num_samples': NUM_TRAINING_SAMPLES, 'seplvl': TASK_SEP, 'seed': RND_SEED}, 
        2, 
        FIXED_LENGTHS, 
        NUM_FOLDS, 
        fsuffix,
        RND_SEED,
        show_result)
          
           
    run_single_experiment(
        MethodNN, 
        {'distfcn': 'cosine', 'stride': 3},
        SyntheticData, 
        {'num_samples': NUM_TRAINING_SAMPLES, 'seplvl': TASK_SEP, 'seed': RND_SEED},
        2, 
        FIXED_LENGTHS, 
        NUM_FOLDS, 
        fsuffix,
        RND_SEED,
        show_result)
          
          
    run_single_experiment(
        MethodLSTM, 
        {'max_seq_len': 20, 'strat': 'sequential', 'epochs': 20, 'fname': 'lstm_model'}, 
        SyntheticData, 
        {'num_samples': NUM_TRAINING_SAMPLES, 'seplvl': TASK_SEP, 'seed': RND_SEED},
        2, 
        FIXED_LENGTHS, 
        NUM_FOLDS, 
        fsuffix,
        RND_SEED,
        show_result)



if __name__ == '__main__':
    main()
    
    