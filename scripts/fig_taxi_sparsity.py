'''
'''

import matplotlib.pyplot as plt
from helpers import *

from exp_taxi_sparse import *

from run_experiment import read_result, comp_mean_ci, FIGURES_DIR


NUM_SAMPLES = range(200, 1001, 200) 

# load all results
markov_all = np.zeros((NUM_FOLDS, 2, len(NUM_SAMPLES)))
bbcf_all = np.zeros((NUM_FOLDS, 2, len(NUM_SAMPLES)))
nncos_all = np.zeros((NUM_FOLDS, 2, len(NUM_SAMPLES)))
nneuc_all = np.zeros((NUM_FOLDS, 2, len(NUM_SAMPLES)))
lstm_all = np.zeros((NUM_FOLDS, 2, len(NUM_SAMPLES)))

for ids, ns in enumerate(NUM_SAMPLES):
    tmp, _ = read_result(
        MethodMarkov, 
        {'reg_factor': REGULARIZATION_FACTOR}, 
        SparseTaxiData, 
        {'num_samples': LOAD_SIZE, 'num_train': ns, 'num_test': TEST_SIZE, 'num_goals': NUM_GOALS, 'grid': [20, 20]},
        fsuffix)
    markov_all[:,:,ids] = tmp


    tmp, _ = read_result(
        MethodBBCF,
        {'reg_factor': REGULARIZATION_FACTOR, 'num_models': NUM_GOALS},
        SparseTaxiData, 
        {'num_samples': LOAD_SIZE, 'num_train': ns, 'num_test': TEST_SIZE, 'num_goals': NUM_GOALS, 'grid': [20, 20]},
        fsuffix)
    bbcf_all[:,:,ids] = tmp
      
       
    tmp, _ = read_result(
        MethodNN, 
        {'distfcn': 'euclidean', 'stride': 3},
        SparseTaxiData, 
        {'num_samples': LOAD_SIZE, 'num_train': ns, 'num_test': TEST_SIZE, 'num_goals': NUM_GOALS, 'grid': [20, 20]},
        fsuffix)
    nneuc_all[:,:,ids] = tmp
        
         
    tmp, _ = read_result(
        MethodNN, 
        {'distfcn': 'cosine', 'stride': 3},
        SparseTaxiData, 
        {'num_samples': LOAD_SIZE, 'num_train': ns, 'num_test': TEST_SIZE, 'num_goals': NUM_GOALS, 'grid': [20, 20]},
        fsuffix)
    nncos_all[:,:,ids] = tmp
    
        
    tmp, _ = read_result(
        MethodLSTM, 
        {'max_seq_len': 10, 'strat': 'sequential', 'epochs': 20, 'fname': 'lstm_model'}, 
        SparseTaxiData, 
        {'num_samples': LOAD_SIZE, 'num_train': ns, 'num_test': TEST_SIZE, 'num_goals': NUM_GOALS, 'grid': [20, 20]},
        fsuffix)
    lstm_all[:,:,ids] = tmp
    
    
markov_mean = np.zeros((2, len(NUM_SAMPLES)))
markov_conf = np.zeros((2, len(NUM_SAMPLES)))
bbcf_mean = np.zeros((2, len(NUM_SAMPLES)))
bbcf_conf = np.zeros((2, len(NUM_SAMPLES)))
nncos_mean = np.zeros((2, len(NUM_SAMPLES)))
nncos_conf = np.zeros((2, len(NUM_SAMPLES)))
nneuc_mean = np.zeros((2, len(NUM_SAMPLES)))
nneuc_conf = np.zeros((2, len(NUM_SAMPLES)))
lstm_mean = np.zeros((2, len(NUM_SAMPLES)))
lstm_conf = np.zeros((2, len(NUM_SAMPLES)))

for hr in range(2):
    for ns in range(len(NUM_SAMPLES)):
        markov_mean[hr,ns], markov_conf[hr,ns] = comp_mean_ci(markov_all[:,hr,ns])
        bbcf_mean[hr,ns], bbcf_conf[hr,ns] = comp_mean_ci(bbcf_all[:,hr,ns])
        nncos_mean[hr,ns], nncos_conf[hr,ns] = comp_mean_ci(nncos_all[:,hr,ns])
        nneuc_mean[hr,ns], nneuc_conf[hr,ns] = comp_mean_ci(nneuc_all[:,hr,ns])
        lstm_mean[hr,ns], lstm_conf[hr,ns] = comp_mean_ci(lstm_all[:,hr,ns])


# accuracy plots
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 12})

for hr in range(2):
    plt.figure()
    plt.plot(NUM_SAMPLES, markov_mean[hr], 'r')
    plt.plot(NUM_SAMPLES, nncos_mean[hr], 'g')
    plt.plot(NUM_SAMPLES, nneuc_mean[hr], 'm')
    plt.plot(NUM_SAMPLES, lstm_mean[hr], 'k')
    plt.plot(NUM_SAMPLES, bbcf_mean[hr], 'b')
    
    plt.fill_between(NUM_SAMPLES, markov_mean[hr] - markov_conf[hr], markov_mean[hr] + markov_conf[hr], alpha=0.5, edgecolor='r', facecolor='r', linewidth=0)
    plt.fill_between(NUM_SAMPLES, nncos_mean[hr] - nncos_conf[hr], nncos_mean[hr] + nncos_conf[hr], alpha=0.5, edgecolor='g', facecolor='g', linewidth=0)
    plt.fill_between(NUM_SAMPLES, nneuc_mean[hr] - nneuc_conf[hr], nneuc_mean[hr] + nneuc_conf[hr], alpha=0.5, edgecolor='m', facecolor='m', linewidth=0)
    plt.fill_between(NUM_SAMPLES, lstm_mean[hr] - lstm_conf[hr], lstm_mean[hr] + lstm_conf[hr], alpha=0.5, edgecolor='k', facecolor='k', linewidth=0)
    plt.fill_between(NUM_SAMPLES, bbcf_mean[hr] - bbcf_conf[hr], bbcf_mean[hr] + bbcf_conf[hr], alpha=0.5, edgecolor='b', facecolor='b', linewidth=0)
      
    plt.title('Taxi Navigation Recommendation')
    if hr == 0:
        plt.ylabel('accuracy (%)')
    else:
        plt.ylabel('hit rate @ 2 (%)')
    plt.xticks(NUM_SAMPLES)
    plt.xlabel('number of train samples')
    plt.legend(['Markov', 'NNcos', 'NNeuc', 'LSTM', 'BBCF'])

    figname = os.path.join(FIGURES_DIR, 'taxi_sparsity_hr' + str(hr+1) + '_comparison_grid{:d}-{:d}'.format(GRID_SIZE[0], GRID_SIZE[1]) + '.pdf')
    ensure_directory(figname, True)
    plt.savefig(figname)


