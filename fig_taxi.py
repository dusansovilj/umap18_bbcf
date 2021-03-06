'''
'''

import matplotlib.pyplot as plt
from exp_taxi import *
from helpers.aux import *
from helpers.run_experiment import read_result, comp_mean_ci_4lengths, FIGURES_DIR

# load all results
markov_full, markov_len = read_result(
    MethodMarkov, 
    {'reg_factor': REGULARIZATION_FACTOR}, 
    TaxiData, 
    {'num_samples': TRAIN_SIZE, 'grid': GRID_SIZE},
    fsuffix)
markov_mean, markov_conf = comp_mean_ci_4lengths(100 * markov_len)


bbcf_full, bbcf_len = read_result(
    MethodBBCF,
    {'reg_factor': REGULARIZATION_FACTOR, 'num_models': NUM_GOALS},
    TaxiData, 
    {'num_samples': TRAIN_SIZE, 'grid': GRID_SIZE},
    fsuffix)
bbcf_mean, bbcf_conf = comp_mean_ci_4lengths(100 * bbcf_len)

 
nneuc_full, nneuc_len = read_result(
    MethodNN, 
    {'distfcn': 'euclidean', 'stride': 5},
    TaxiData, 
    {'num_samples': TRAIN_SIZE, 'grid': GRID_SIZE},
    fsuffix)
nneuc_mean, nneuc_conf = comp_mean_ci_4lengths(100 * nneuc_len)
 
  
nncos_full, nncos_len = read_result(
    MethodNN, 
    {'distfcn': 'cosine', 'stride': 5},
    TaxiData, 
    {'num_samples': TRAIN_SIZE, 'grid': GRID_SIZE},
    fsuffix)
nncos_mean, nncos_conf = comp_mean_ci_4lengths(100 * nncos_len)
 
 
lstm_full, lstm_len = read_result(
    MethodLSTM, 
    {'max_seq_len': 10, 'strat': 'sequential', 'epochs': 20, 'fname': 'lstm_model'}, 
    TaxiData, 
    {'num_samples': TRAIN_SIZE, 'grid': GRID_SIZE},
    fsuffix)
lstm_mean, lstm_conf = comp_mean_ci_4lengths(100 * lstm_len)


# accuracy plots
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 12})

for hr in range(2):
    plt.figure()
    plt.plot(FIXED_LENGTHS, markov_mean[hr], 'r')
    plt.plot(FIXED_LENGTHS, nncos_mean[hr], 'g')
    plt.plot(FIXED_LENGTHS, nneuc_mean[hr], 'm')
    plt.plot(FIXED_LENGTHS, lstm_mean[hr], 'k')
    plt.plot(FIXED_LENGTHS, bbcf_mean[hr], 'b')
      
    plt.fill_between(FIXED_LENGTHS, markov_mean[hr] - markov_conf[hr], markov_mean[hr] + markov_conf[hr], alpha=0.5, edgecolor='r', facecolor='r', linewidth=0)
    plt.fill_between(FIXED_LENGTHS, nncos_mean[hr] - nncos_conf[hr], nncos_mean[hr] + nncos_conf[hr], alpha=0.5, edgecolor='g', facecolor='g', linewidth=0)
    plt.fill_between(FIXED_LENGTHS, nneuc_mean[hr] - nneuc_conf[hr], nneuc_mean[hr] + nneuc_conf[hr], alpha=0.5, edgecolor='m', facecolor='m', linewidth=0)
    plt.fill_between(FIXED_LENGTHS, lstm_mean[hr] - lstm_conf[hr], lstm_mean[hr] + lstm_conf[hr], alpha=0.5, edgecolor='k', facecolor='k', linewidth=0)
    plt.fill_between(FIXED_LENGTHS, bbcf_mean[hr] - bbcf_conf[hr], bbcf_mean[hr] + bbcf_conf[hr], alpha=0.5, edgecolor='b', facecolor='b', linewidth=0)
      
    plt.title('Taxi Navigation Recommendation')
    if hr == 0:
        plt.ylabel('accuracy (%)')
    else:
        plt.ylabel('hit rate @ 2 (%)')
    plt.xlabel('sequence length')
    plt.legend(['Markov', 'NNcos', 'NNeuc', 'LSTM', 'BBCF'], loc='lower right')
      
    figname = os.path.join(FIGURES_DIR, 'taxi_hr' + str(hr+1) + '_comparison_' + fsuffix + '.pdf')
    ensure_directory(figname, True)
    plt.savefig(figname)


