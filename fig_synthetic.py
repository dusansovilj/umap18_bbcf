'''
'''

import matplotlib.pyplot as plt
import numpy as np
from helpers.aux import *
from helpers.run_experiment import FIGURES_DIR, comp_mean_ci, read_result
from exp_synthetic import *


SEPARABILITY_LEVELS = [0., 0.2, 0.4, 0.6, 0.8, 1.]

# load all results
markov_all = np.zeros((NUM_FOLDS, 2, len(SEPARABILITY_LEVELS)))
bbcf_all   = np.zeros((NUM_FOLDS, 2, len(SEPARABILITY_LEVELS)))
nncos_all  = np.zeros((NUM_FOLDS, 2, len(SEPARABILITY_LEVELS)))
nneuc_all  = np.zeros((NUM_FOLDS, 2, len(SEPARABILITY_LEVELS)))
lstm_all   = np.zeros((NUM_FOLDS, 2, len(SEPARABILITY_LEVELS)))

for ids, ts in enumerate(SEPARABILITY_LEVELS):
    tmp, _ = read_result(
        MethodMarkov, 
        {'reg_factor': REGULARIZATION_FACTOR}, 
        SyntheticData, 
        {'num_samples': NUM_TRAINING_SAMPLES, 'seplvl': ts, 'seed': RND_SEED},
        fsuffix)
    markov_all[:,:,ids] = tmp


    tmp, _ = read_result(
        MethodBBCF, 
        {'reg_factor': REGULARIZATION_FACTOR, 'num_models': NUM_TASKS},
        SyntheticData, 
        {'num_samples': NUM_TRAINING_SAMPLES, 'seplvl': ts, 'seed': RND_SEED},
        fsuffix)
    bbcf_all[:,:,ids] = tmp
      
       
    tmp, _ = read_result(
        MethodNN, 
        {'distfcn': 'euclidean', 'stride': 3},
        SyntheticData, 
        {'num_samples': NUM_TRAINING_SAMPLES, 'seplvl': ts, 'seed': RND_SEED},
        fsuffix)
    nneuc_all[:,:,ids] = tmp
        
         
    tmp, _ = read_result(
        MethodNN, 
        {'distfcn': 'cosine', 'stride': 3},
        SyntheticData, 
        {'num_samples': NUM_TRAINING_SAMPLES, 'seplvl': ts, 'seed': RND_SEED},
        fsuffix)
    nncos_all[:,:,ids] = tmp
    
        
    tmp, _ = read_result(
        MethodLSTM, 
        {'max_seq_len': 20, 'strat': 'sequential', 'epochs': 20, 'fname': 'lstm_model'}, 
        SyntheticData, 
        {'num_samples': NUM_TRAINING_SAMPLES, 'seplvl': ts, 'seed': RND_SEED},
        fsuffix)
    lstm_all[:,:,ids] = tmp
    

markov_mean = np.zeros((2, len(SEPARABILITY_LEVELS)))
markov_conf = np.zeros((2, len(SEPARABILITY_LEVELS)))
bbcf_mean   = np.zeros((2, len(SEPARABILITY_LEVELS)))
bbcf_conf   = np.zeros((2, len(SEPARABILITY_LEVELS)))
nncos_mean  = np.zeros((2, len(SEPARABILITY_LEVELS)))
nncos_conf  = np.zeros((2, len(SEPARABILITY_LEVELS)))
nneuc_mean  = np.zeros((2, len(SEPARABILITY_LEVELS)))
nneuc_conf  = np.zeros((2, len(SEPARABILITY_LEVELS)))
lstm_mean   = np.zeros((2, len(SEPARABILITY_LEVELS)))
lstm_conf   = np.zeros((2, len(SEPARABILITY_LEVELS)))

for hr in range(2):
    for sl in range(len(SEPARABILITY_LEVELS)):
        markov_mean[hr,sl], markov_conf[hr,sl] = comp_mean_ci(markov_all[:,hr,sl])
        bbcf_mean[hr,sl], bbcf_conf[hr,sl] = comp_mean_ci(bbcf_all[:,hr,sl])
        nncos_mean[hr,sl], nncos_conf[hr,sl] = comp_mean_ci(nncos_all[:,hr,sl])
        nneuc_mean[hr,sl], nneuc_conf[hr,sl] = comp_mean_ci(nneuc_all[:,hr,sl])
        lstm_mean[hr,sl], lstm_conf[hr,sl] = comp_mean_ci(lstm_all[:,hr,sl])

 
# accuracy plots
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 12})

for hr in range(2):
    plt.figure()
    plt.plot(SEPARABILITY_LEVELS, markov_mean[hr], 'r')
    plt.plot(SEPARABILITY_LEVELS, nncos_mean[hr], 'g')
    plt.plot(SEPARABILITY_LEVELS, nneuc_mean[hr], 'm')    
    plt.plot(SEPARABILITY_LEVELS, lstm_mean[hr], 'k')
    plt.plot(SEPARABILITY_LEVELS, bbcf_mean[hr], 'b')
    
    plt.fill_between(SEPARABILITY_LEVELS, markov_mean[hr] - markov_conf[hr], markov_mean[hr] + markov_conf[hr], alpha=0.5, edgecolor='r', facecolor='r', linewidth=0)
    plt.fill_between(SEPARABILITY_LEVELS, nncos_mean[hr] - nncos_conf[hr], nncos_mean[hr] + nncos_conf[hr], alpha=0.5, edgecolor='g', facecolor='g', linewidth=0)
    plt.fill_between(SEPARABILITY_LEVELS, nneuc_mean[hr] - nneuc_conf[hr], nneuc_mean[hr] + nneuc_conf[hr], alpha=0.5, edgecolor='m', facecolor='m', linewidth=0)
    plt.fill_between(SEPARABILITY_LEVELS, lstm_mean[hr] - lstm_conf[hr], lstm_mean[hr] + lstm_conf[hr], alpha=0.5, edgecolor='k', facecolor='k', linewidth=0)
    plt.fill_between(SEPARABILITY_LEVELS, bbcf_mean[hr] - bbcf_conf[hr], bbcf_mean[hr] + bbcf_conf[hr], alpha=0.5, edgecolor='b', facecolor='b', linewidth=0)
       
    plt.title('Synthetic Sequential Prediction')
    if hr == 0:
        plt.ylabel('accuracy (%)')
    else:
        plt.ylabel('hit rate @ 2 (%)')
    plt.xlabel('separability level')
    plt.legend(['Markov', 'NNcos', 'NNeuc', 'LSTM', 'BBCF'])
    
    figname = os.path.join(FIGURES_DIR, 'synthetic_separation_hr' + str(hr+1) + '.pdf')
    ensure_directory(figname, True)
    plt.savefig(figname)


