'''
Created on Mar 2, 2018
'''

import matplotlib.pyplot as plt
from helpers import *


# FIXED_LENGTHS = [2, 3, 4, 5, 6, 7, 8, 9, 10]


NUM_FOLDS = 10
SKILL_SET = 3
NUM_NEIGHS = 11
MAX_HIT_RATE = 1

# fsuffix = 'skillset{:d}_nns{:d}'.format(SKILL_SET, NUM_NEIGHS)
# 
# markov_res_file = 'results/kt/kt_markov_' + fsuffix
# cfirl_res_file = 'results/taxi/kt_cfirl_' + fsuffix
# nn_res_file = 'results/kt/kt_nn_' + fsuffix
# lstm_res_file = 'results/kt/kt_lstm_' + fsuffix
# 
# 
# # load all results
# nncos_all = np.zeros((NUM_FOLDS, MAX_HIT_RATE))
# nneuc_all = np.zeros((NUM_FOLDS, MAX_HIT_RATE))
#  
# for fold in range(NUM_FOLDS):
#     fname = nn_res_file +  '_cosine_fold' + str(fold + 1)
#     [v1, v2] = load_res_file(fname)
#     if v1 != None and v2 != None:
#         nncos_all[fold] = v1
#           
#     fname = nn_res_file +  '_euclidean_fold' + str(fold + 1)
#     [v1, v2] = load_res_file(fname)
#     if v1 != None and v2 != None:
#         nneuc_all[fold] = v1
#      
#  
#  
# print('\n{:10s}{:8s}{:>10s}{:>10s}\n{:s}'.format('model', 'length', 'accuracy', 'acc conf', ''.join(40 * ['-'])))
# print('{:s}'.format(''.join(40 * ['-'])))
#  
#  
#  
# print('{:10s}{:>8s}{:10.2f}{:10.2f}'.format('nncos', 'all', 100 * np.mean(nncos_all[:,0]), 100 * comp_95confint(nncos_all[:,0])))
# print('{:s}\n'.format(''.join(40 * ['-'])))
#  
#  
# print('{:10s}{:>8s}{:10.2f}{:10.2f}'.format('nneuc', 'all', 100 * np.mean(nneuc_all[:,0]), 100 * comp_95confint(nneuc_all[:,0])))



# # user modelling 
#  
# nn_res_file = 'results/kt/users/kt_user_nn_'
#  
# NUM_USERS = [156, 297, 128, 561]
#  
# nncos_all = []
# nneuc_all = []
#  
# for user in range(NUM_USERS[SKILL_SET]):
#     for fold in range(NUM_FOLDS):
#         try:
#             fname = nn_res_file + 'userid{:d}_skillset{:d}_nns{:d}_cosine_fold{:d}'.format(user, SKILL_SET, NUM_NEIGHS, fold+1)
#             [v1, v2] = load_res_file(fname)
#             if v1 != None and v2 != None:
#                 nncos_all.append(v1)
#         except:
#             pass
#              
#              
#         try:
#             fname = nn_res_file + 'userid{:d}_skillset{:d}_nns{:d}_euclidean_fold{:d}'.format(user, SKILL_SET, NUM_NEIGHS, fold+1)
#             [v1, v2] = load_res_file(fname)
#             if v1 != None and v2 != None:
#                 nneuc_all.append(v1)
#         except:
#             pass
#          
#  
# print('\n{:10s}{:8s}{:>10s}{:>10s}\n{:s}'.format('model', 'length', 'accuracy', 'acc conf', ''.join(40 * ['-'])))
# print('{:s}'.format(''.join(40 * ['-'])))
#   
#  
# print('{:10s}{:>8s}{:10.2f}{:10.2f}'.format('nncos', 'all', 100 * np.mean(nncos_all), 100 * comp_95confint(np.array(nncos_all))))
# print('{:s}\n'.format(''.join(40 * ['-'])))
#   
#  
# print('{:10s}{:>8s}{:10.2f}{:10.2f}'.format('nneuc', 'all', 100 * np.mean(nneuc_all), 100 * comp_95confint(np.array(nneuc_all))))


