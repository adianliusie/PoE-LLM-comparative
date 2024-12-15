import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from scipy import stats
from collections import Counter, defaultdict
from matplotlib.ticker import FormatStrFormatter

#== Data Analysis Utils =======================================================================================================#
def plot_bivariate_distribution(probs_list:list, dists_list:list):
    sns.kdeplot(x=dists_list, y=probs_list, fill=True)

    plt.xlabel('true score difference')
    plt.ylabel('P(A>B)')

    pcc = float(stats.pearsonr(probs_list, dists_list)[0])
    scc = float(stats.spearmanr(probs_list, dists_list)[0])

    plt.title(f'PCC={pcc:.3f}, SCC={scc:.3f}')

    
def plot_gaussian_cuts(probs_list:list, dists_list:list):
    e = 0.05

    dist_list_by_probs = defaultdict(list)

    for P in np.arange(0,1.1,0.1):
        for prob, dist in zip(probs_list, dists_list):
            if np.abs(P - prob) < e:
                dist_list_by_probs[round(P, 1)].append(dist)
    
    min_dist = min(dists_list)-0.3
    max_dist = max(dists_list)+0.3
    
    #y_lim = max([Counter(v).most_common(1)[0][1]/len(v) for v in dist_list_by_probs.values()])+0.03
    #y_lim = 0.2
    
    # Plot the sorted PMF as a bar chart
    matplotlib.rcdefaults()
    fig, axs = plt.subplots(2, 3, figsize=(15, 6))


    for k, P in enumerate([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
        ax = axs[k//3, k%3]
        weight = np.ones(len(dist_list_by_probs[P])) / len(dist_list_by_probs[P])
        binwidth= 1/3
        bins = np.arange(min(dist_list_by_probs[P])-binwidth/2, max(dist_list_by_probs[P])+binwidth/2 + binwidth, binwidth)

        ax.hist(dist_list_by_probs[P], bins=bins, edgecolor='black', linewidth=1.2, color='skyblue', weights=weight)
        ax.set_xlim(min_dist, max_dist)
        #ax.set_ylim(0, y_lim)
        ax.set_title(f"P(A>B) = {P},  $\\mu$={np.mean(dist_list_by_probs[P]):.2f},  $\\sigma$={np.std(dist_list_by_probs[P]):.2f}")
        ax.axvline(0, linestyle='--', color='k', alpha=0.4)

    plt.subplots_adjust(hspace=0.3, wspace=0.2)

def plot_mean_std_by_probs(probs_list: list, dists_list: list):
    sns.set(style="ticks", palette="muted")  # Use a minimalistic and professional style

    e = 0.005  # Proximity threshold for matching probabilities
    dist_list_by_probs = defaultdict(list)

    # Organize distributions by probabilities within the threshold e
    for P in np.arange(0, 1.01, 0.01):
        for prob, dist in zip(probs_list, dists_list):
            if np.abs(P - prob) < e:
                dist_list_by_probs[round(P, 3)].append(dist)

    # Calculate mean and standard deviation for each probability bucket
    mean_05 = [np.mean(x) if x else 0 for x in dist_list_by_probs.values()]
    std_05 = [np.std(x) if x else 0 for x in dist_list_by_probs.values()]

    # Prepare the 'counts' data to overlay as a bar graph for context
    counts = np.array([len(x) for x in dist_list_by_probs.values()])
    k = max(max(mean_05, default=0), max(std_05, default=0)) / max(counts, default=1)
    counts_scaled = k * counts

    x_axis = list(dist_list_by_probs.keys())

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis

    # Plotting the standard deviation
    color = 'tab:blue'
    #ax1.set_ylabel('Mean', color=color, fontsize=14)
    ax1.plot(x_axis, mean_05, label='Mean ($\mu$)', color=color)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=14)

    color = 'tab:red'
    ax2.set_xlabel('Probability P(A>B)', fontsize=14)
    #ax2.set_ylabel('Standard Deviation', color=color, fontsize=14)
    ax2.plot(x_axis, std_05, label='Standard Deviation ($\sigma$)', color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=14)

    # Plot the counts as a filled area for better visibility
    ax2.fill_between(x_axis, 0, counts_scaled, color='gray', alpha=0.15, label='Scaled Frequency')
    ax1.tick_params(axis='x', labelsize=14)  # Increase x-axis tick label size

    # Improve the legend with better placement and font size
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='lower right', fontsize='large')
    
    plt.xlim(0, 1)  # Ensure the x-axis goes from 0 to 1
    ax2.set_ylim(0)
    
    plt.grid(True)  # Add grid for better readability
    
    
#== Plotting Efficiency Curves ================================================================================================#
def plot_efficiency_curves(results:dict, metric='scc'):
    assert metric in ['spearman', 'pearson']
    
    if metric == 'spearman':
        wr = [results[k]['wr-scc'] for k in results]
        probs = [results[k]['prob-scc'] for k in results]
        mle = [results[k]['mle-scc'] for k in results]
        mle_hard = [results[k]['mle-hard-scc'] for k in results]
        y_label = 'spearman'
        
    elif metric == 'pearson':
        wr = [results[k]['wr-pcc'] for k in results]
        probs = [results[k]['prob-pcc'] for k in results]
        mle = [results[k]['mle-pcc'] for k in results]
        mle_hard = [results[k]['mle-hard-pcc'] for k in results]
        y_label = 'pearson'

    
    x_axis = [k for k in results]
    
    # Use seaborn style for aesthetics
    sns.set(style="whitegrid")
    
    plt.figure(figsize=(7, 5))
    plt.plot(x_axis, mle, label='MLE-POE probs', marker='d', linestyle='-', color='red')
    plt.plot(x_axis, probs, label='avg-prob', marker='s', linestyle='-', color='blue')
    plt.plot(x_axis, mle_hard, label='MLE-POE dec', marker='^', linestyle='--', color='red')
    plt.plot(x_axis, wr, label='win-ratio', marker='o', linestyle='--', color='green')

    # Labels and title
    plt.xlabel('Number of Comparisons', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    #plt.title(f'Efficient Comparison Curves', fontsize=14)
    
    # Legend
    plt.legend(title='comparisons to scores', fancybox=True, shadow=True)
    
    # Enhancements
    plt.xticks(x_axis, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()

def plot_efficiency_curves_with_error_bars(
    data:dict, 
    methods:list=None #['wr-scc', 'prob-scc', 'mle-hard-scc', 'mle-scc']
):
    sns.set(style="whitegrid")
    all_sets = [(int(R/2), method, v) for R, methods in data.items() for method, scores in methods.items() for v in scores]

    y_label = 'spearman' if all_sets[0][1].endswith('scc') else 'pearson'
    
    if methods is None:
        filtered_sets = all_sets
    else:
        filtered_sets = [x for x in all_sets if x[1] in methods]
    
    df = pd.DataFrame(filtered_sets, columns=['R', 'Method', 'Score'])
    
    sns.pointplot(data=df, x='R', y='Score', hue='Method', capsize=.2, markers=["o", "^", "v", "s", "+", "."])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.xlabel('Number of Comparisons', fontsize=12)
    plt.ylabel(y_label, fontsize=12)

def plot_efficiency_curves_new(data:dict, methods:list=None, log=False, half=False):
    def calculate_confidence_interval(values, confidence=0.95):     # 95% Confidence Interval function
        sem = np.std(values, ddof=1) / np.sqrt(len(values)) # calculate standard error
        h = sem * stats.t.ppf((1 + confidence) / 2., len(values) - 1)
        return h
    
    x_axis = [k for k in data]
    
    if half:
        x_axis = [int(k/2) for k in data] # because in symemtric we're using num comparisons

    sns.set(style="whitegrid")
    plt.figure(figsize=(7, 5))

    markers = ["o", "^", "v", "s", "+", "."]
    for k, method in enumerate(methods[::-1]):
        y_means = np.array([np.mean(data[R][method]) for R in x_axis])
        y_std = np.array([calculate_confidence_interval(data[R][method]) for R in x_axis])
        
        label = method.replace('-scc', '').replace('-pcc', '')
        if   label == 'zermelo-hard' : label = 'BT'
        elif label == 'gaussian-hard': label = 'POE-g-hard'
        elif label == 'gaussian'     : label = 'POE-g'
        elif label == 'sigmoid'      : label = 'POE-BT'
            
        plt.plot(x_axis, y_means, label=label, marker=markers[k], linestyle='-', linewidth=2)
        
        # now plot error bars around the mean with +- std
        plt.fill_between(x_axis, y_means-y_std, y_means+y_std, alpha=0.2)
        
    y_label = 'pearson' if methods[0].endswith('pcc') else 'spearman'
    plt.xlabel('Number of Comparisons', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    if log:
        plt.xscale('log')
        formatter = FormatStrFormatter('%.d')
        plt.gca().xaxis.set_major_formatter(formatter)

        tick_values = [2000, 5000, 10000, 20000, 50000, 100000]  # Modify `num` for more/less ticks
        plt.xticks(tick_values, fontsize=10)

    # Legend
    plt.legend(title='method', fancybox=True, shadow=True)
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    
def plot_efficiency_curves_final(data:dict, methods:list=None, log=False, half=False):
    def calculate_confidence_interval(values, confidence=0.95):     # 95% Confidence Interval function
        sem = np.std(values, ddof=1) / np.sqrt(len(values)) # calculate standard error
        h = sem * stats.t.ppf((1 + confidence) / 2., len(values) - 1)
        return h
    
    x_axis = [k for k in data]
    
    if half:
        x_axis = [int(k/2) for k in data] # because in symemtric we're using num comparisons

    sns.set(style="whitegrid")
    plt.figure(figsize=(5.5, 4))
    
    styles = {
        'win-ratio': ['win-ratio', (0, (5,1)), '#99c2ff', 'o'],  # Deep Blue Dark
        'zermelo-hard': ['Bradley-Terry', (0, (5,1)), '#ffad66', '^'],  # Deep Orange Light
        'gaussian-hard': ['PoE-gaussian-hard', (0, (5,1)), '#85e085', '.'],  # Deep Green Light
        'avg-prob': ['average-probability', '-', '#336699', 's'],  # Deep Blue Light
        's-zermelo': ['PoE-Bradley-Terry', '-', '#e67300', '+'],  # Deep Orange Dark
        'gaussian': ['PoE-gaussian', '-', '#348017', 'v'],  # Deep Green Dark
        
        'bt-gaussian-approx': ['PoE-gaussian-BT-approx', '-', 'red', 'v'],  # Deep Green Dark

        'gaussian': ['PoE-g', '-.', '#348017', '+'],  # Deep Green Dark
        'gaussian-d': ['PoE-g-debiased', '-', '#85e085', 'v'],  # Deep Green Dark
        's-zermelo': ['PoE-BT', '-.', '#e67300', '+'],  # Deep Orange Dark
        'sigmoid-d': ['PoE-BT-debiased', '-', '#ffad66', 'v'],  # Deep Green Dark

        
        'optimal gaussian': ['selected PoE-gaussian', '-', '#348017', 'v'],
        'random gaussian': ['random PoE-gaussian', '-.', '#85e085', 'v'],
        'optimal sigmoid': ['selected PoE-Bradley-Terry', '-', '#e67300', 'v'],
        'random sigmoid': ['random PoE-Bradley-Terry', '-.', '#ffad66', 'v'],

        'balanced PoE-g': ['symmetric PoE-g', '-', '#348017', 'o'],
        'unbalanced PoE-g': ['non-symmetric PoE-g', '-.', '#85e085', '^'],
        'balanced PoE-BT': ['symmetric PoE-BT', '-', '#e67300', '.'],
        'unbalanced PoE-BT': ['non-symmetric PoE-BT', '-.', '#ffad66', 's'],
        'balanced avg-prob': ['symmetric avg-prob', '-', '#336699', '+'],
        'unbalanced avg-prob': ['non-symmetric avg-prob', '-.', '#99c2ff', 'v'],
        
        'debug': ['debug', '-', '#ffad66', 'v'],  # Deep Green Dark

    }
    
    #     styles = {
    #         'win-ratio': ['win-ratio', (0, (5,1)), '#8ecae6', 'o'],  # Deep Blue Dark
    #         'zermelo-hard': ['Bradley-Terry', (0, (5,1)), '#fb6f92', '^'],  # Deep Orange Light
    #         'gaussian-hard': ['PoE-gaussian-hard', (0, (5,1)), '#85e085', '.'],  # Deep Green Light
    #         'avg-prob': ['average-probability', '-', '#219ebc', 's'],  # Deep Blue Light
    #         's-zermelo': ['PoE-Bradley-Terry', '-', '#d00000', '+'],  # Deep Orange Dark
    #         'gaussian': ['PoE-gaussian', '-', '#348017', 'v']  # Deep Green Dark
    #     }


    print(methods)
    for method in methods[::-1]: # reverse order of methods list
        y_means = np.array([np.mean(data[R][method]) for R in x_axis])
        y_std = np.array([calculate_confidence_interval(data[R][method]) for R in x_axis])
        #y_std = np.array([np.std(data[R][method]) for R in x_axis])

        code = method.replace('-scc', '').replace('-pcc', '')
        if code in styles:
            label, linestyle, color, marker = styles[code]
            plt.plot(x_axis, y_means, label=label, marker=marker, linestyle=linestyle, color=color, linewidth=2)
            # now plot error bars around the mean with +- std
            plt.fill_between(x_axis, y_means-y_std, y_means+y_std, color=color, alpha=0.15)

        else:
            plt.plot(x_axis, y_means, linewidth=2)
        
    y_label = 'pearson' if methods[0].endswith('pcc') else 'spearman'
    plt.xlabel('Number of Comparisons', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    if log:
        plt.xscale('log')
        formatter = FormatStrFormatter('%.d')
        plt.gca().xaxis.set_major_formatter(formatter)

        tick_values = [2000, 5000, 10000, 20000, 50000, 100000]  # Modify `num` for more/less ticks
        plt.xticks(tick_values, fontsize=10)
    
    if max(data.keys())<40:
        R_list = list(data.keys())
        x_start = 5 * (min(R_list) // 5)
        x_end = 5 * ((max(R_list) // 5) + 1)
        plt.xticks(np.arange(x_start, x_end + 1, 5))
        plt.xlim(11.5,30.5)

    # Legend
    plt.legend(title='method', fancybox=True, shadow=True)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
