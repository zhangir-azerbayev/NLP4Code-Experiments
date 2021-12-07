import matplotlib.pyplot as plt 
import scipy 
import json 
from scipy import stats 

with open('scores.json') as f: 
    score_list = json.load(f) 

# Sequence length
bin_means, bin_edges, binnumber = stats.binned_statistic([instance["len_score"] for instance in score_list], 
                       [instance["pass_k"] for instance in score_list], 
                       statistic='mean', 
                       bins=10
                      )
plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=7, 
           label='binned statistic of data')
plt.xlabel('Sequence Length')
plt.ylabel('Average pass@10')
plt.savefig('length_hist.pdf')


# Operation score 
plt.close()
bin_means, bin_edges, binnumber = stats.binned_statistic([instance["operation_score"] for instance in score_list], 
                       [instance["pass_k"] for instance in score_list], 
                       statistic='mean', 
                       bins=10
                      )
plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=7, 
           label='binned statistic of data')
plt.xlabel('Operation Score')
plt.ylabel('Average pass@10')
plt.savefig('operation_hist.pdf')


# Unique Operation Score
plt.close()
bin_means, bin_edges, binnumber = stats.binned_statistic([instance["unique_operation_score"] for instance in score_list], 
                       [instance["pass_k"] for instance in score_list], 
                       statistic='mean', 
                       bins=10
                      )
plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=7, 
           label='binned statistic of data')
plt.xlabel('Unique Operation Score')
plt.ylabel('Average pass@10')
plt.savefig('unique_operation_hist.pdf')
