'''
Experiments on Knowledge-Tracing data for nearest-neighbor and neural network.
'''


from datasets.kt import KTData

from run_experiment import run_single_experiment
from algs.nearestneigh import MethodNN
from algs.lstm import MethodLSTM


NUM_FOLDS = 10
SKILL_SET = 2          # change this to 0, 1, 2 or 3 for different KT skill sets
FIXED_LENGTHS = [10]
STRIDE = [2, 2, 2, 5]  # for different skills sets since the last one is larger than others
RND_SEED = 0


fsuffix = 'seed{:d}'.format(SKILL_SET, STRIDE[SKILL_SET], RND_SEED)
show_result = True


def main():
    run_single_experiment(
        MethodNN,
        {'distfcn': 'euclidean', 'stride': STRIDE[SKILL_SET]},
        KTData,
        {'skill_set': SKILL_SET, 'merge': True},
        1, 
        FIXED_LENGTHS, 
        10, 
        fsuffix,
        RND_SEED,
        show_result)
    
    
    run_single_experiment(
        MethodNN,
        {'distfcn': 'cosine', 'stride': STRIDE[SKILL_SET]},
        KTData,
        {'skill_set': SKILL_SET, 'merge': True},
        1, 
        FIXED_LENGTHS, 
        10, 
        fsuffix,
        RND_SEED,
        show_result)
    
    
    run_single_experiment(
        MethodLSTM, 
        {'max_seq_len': 10, 'strat': 'sequential', 'epochs': 20, 'fname': 'lstm_model'}, 
        KTData,
        {'skill_set': SKILL_SET, 'merge': True}, 
        1, 
        FIXED_LENGTHS, 
        10,
        fsuffix, 
        RND_SEED,
        show_result)


if __name__ == '__main__':
    main()
    
    