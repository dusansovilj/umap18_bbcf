'''
Created on Mar 2, 2018

Plotting Harold's results to fit our figure structure
'''

import matplotlib.pyplot as plt
import numpy as np


# mean and std scores extracted from Harold's ipython notebook and pickled file
mean_scores=\
np.array([[ 0.06309335,  0.27495077,  0.25354506],
          [ 0.15754101,  0.27211658,  0.2921743 ],
          [ 0.18324178,  0.26853577,  0.3137637 ],
          [ 0.20228285,  0.26912733,  0.32742298],
          [ 0.21608694,  0.2695633 ,  0.33289143],
          [ 0.23079778,  0.26962548,  0.33358597],
          [ 0.24549513,  0.26849707,  0.34088675],
          [ 0.25277057,  0.26723042,  0.34774699],
          [ 0.26766413,  0.26954735,  0.35518029]])
std_scores=\
np.array([[ 0.01237042,  0.01613866,  0.02174252],
          [ 0.01063476,  0.01543146,  0.02822251],
          [ 0.01328446,  0.01669287,  0.0254984 ],
          [ 0.01490865,  0.01366935,  0.02922605],
          [ 0.01807872,  0.01587377,  0.03677162],
          [ 0.01969454,  0.01943919,  0.0383997 ],
          [ 0.01655211,  0.02125069,  0.03581905],
          [ 0.01756502,  0.01835785,  0.03455832],
          [ 0.02840851,  0.02343652,  0.04058211]])

props = [0.5, 0.2, 0.4, 0.1, 0.6, 0.3, 0.8, 0.9, 0.7]  # props = all_results.keys()
props = np.sort(props)

alglabels = ['Individual', 'Global', 'BBCF']
markers = ['.', 's', 'o']

# accuracy plots
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 12})

plt.figure()
plt.plot(props, mean_scores[:,0], 'r', marker=markers[0])
plt.plot(props, mean_scores[:,1], 'g', marker=markers[1])
plt.plot(props, mean_scores[:,2], 'b', marker=markers[2])

plt.fill_between(props, mean_scores[:,0] - std_scores[:,0], mean_scores[:,0] + std_scores[:,0], alpha=0.5, edgecolor='r', facecolor='r', linewidth=0)
plt.fill_between(props, mean_scores[:,1] - std_scores[:,1], mean_scores[:,1] + std_scores[:,1], alpha=0.5, edgecolor='g', facecolor='g', linewidth=0)
plt.fill_between(props, mean_scores[:,2] - std_scores[:,2], mean_scores[:,2] + std_scores[:,2], alpha=0.5, edgecolor='b', facecolor='b', linewidth=0)


plt.legend(alglabels, loc='lower right')
plt.xlabel("User Data Proportion")
plt.ylabel("MAP")
plt.xlim([0.06, 0.94])
plt.title("MovieLens Collaborative Tagging")
  
plt.savefig('figures/collabtagnb_res_bbcf.pdf')


