import os
import pickle
import numpy as np

from scipy import stats
from typing import List
from collections import defaultdict
from tqdm import tqdm
from functools import lru_cache

from utils import load_comparisons
from utils import win_ratio, avg_prob
from utils import generate_symmetric_random_mask, generate_random_mask
from utils import generate_optimal_symmetric_mask, generate_optimal_mask, random_order_mask
from utils import get_MLE_solution_gaussian, get_MLE_solution_laplace, get_MLE_solution_sigmoid
from utils import get_zermelo_solution, get_soft_zermelo_solution


def get_cache_efficiency_curve(
    dataset:str,
    system:str,
    score_type:str, 
    balanced:str,
    metrics:List[str]=['win-ratio', 'avg-prob', 'gaussian', 'laplace', 'sigmoid'],
    mask:str='random-symmetric',
    start:int=None,
    end:int=None,
    steps:int=None,
    S=100
):  
    assert mask.endswith('symmetric') == balanced

    C_list, labels = load_comparisons(dataset=dataset, system=system, score_type=score_type, balanced=balanced)
    N_ctx, _, N_sys = C_list.shape

    if start is None: start = 2*N_sys 
    if end is None: end = N_sys*(N_sys-1)+1 
    if steps is None: steps = (N_sys if (end-start)/N_sys > 5 else int(N_sys/3)) 

    graph_data = defaultdict(dict)
    raw_data = {}
    
    for R in tqdm(range(start, end, steps)):
        metrics_lists = defaultdict(list)

        for metric in metrics:
            spearman_list, pearson_list = get_cached_metric(metric, dataset, system, score_type, balanced, mask, R, S)
            
            if metric == 'debug': continue
                
            assert len(spearman_list) == len(pearson_list), print(len(spearman_list), len(pearson_list), S*N_ctx)
            assert np.abs(S*N_ctx - len(spearman_list)) <= 0.05*S*N_ctx, print(len(spearman_list), len(pearson_list), S*N_ctx)

            #assert len(spearman_list)%S==0
            if len(spearman_list)%S!=0:
                print('would have failed...')
                
            
            metrics_lists[f"{metric}-scc"] = spearman_list
            metrics_lists[f"{metric}-pcc"] = pearson_list

        graph_data[R] = {k: np.mean(v) for k,v in metrics_lists.items()}
        raw_data[R] = {k:[np.mean(v[i::S]) for i in range(S)] for k,v in metrics_lists.items()}
        
    return graph_data, raw_data


BASE_CACHE_PATH = 'define_path_to_save_cache'

def get_cached_metric(metric:str, dataset:str, system:str, score_type:str, balanced:bool, mask:str, R:int, S:int):
    assert metric in ['win-ratio', 'avg-prob', 'gaussian', 'laplace', 'sigmoid', 'gaussian-hard', 'laplace-hard', 'sigmoid-hard', 'zermelo-hard', 's-zermelo', 'bt-gaussian-approx', 'debug']
    cache_path = f"{BASE_CACHE_PATH}/{dataset}/{score_type}/{system}/{mask}/{metric}/{R}.pk"

    if not os.path.isfile(cache_path):
        print(metric)
        if metric == 'debug': S=5
        C_list, labels = load_comparisons(dataset=dataset, system=system, score_type=score_type, balanced=balanced)
        hard=False
        if metric.endswith('-hard'): 
            metric = metric[:-5]
            hard=True

        dir_path = os.path.dirname(cache_path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        pearson_list, spearman_list = [], []
        for C, ctx_labels in zip(C_list, labels):  
            if hard == True:
                C_hard = (C>0.5).astype(int)
                if np.all(C == C.T):
                    C_hard[np.tril(C)==0.5] = 1 - C_hard.T[np.tril(C)==0.5]
                C = C_hard

            R_count, break_count = 0, 0
            while R_count < S:
                M = generate_mask(mask, R, N=C.shape[-1])
                try:
                    if   metric == 'win-ratio': y_pred = win_ratio(C, M)
                    elif metric == 'avg-prob':  y_pred = avg_prob(C, M)
                    elif metric == 'gaussian':  y_pred = get_MLE_solution_gaussian(C, M)
                    elif metric == 'laplace':   y_pred = get_MLE_solution_laplace(C, M)
                    elif metric == 'sigmoid':   y_pred = get_MLE_solution_sigmoid(C, M)
                    elif metric == 'zermelo':   y_pred = get_zermelo_solution(C, M)
                    elif metric == 's-zermelo': y_pred = get_soft_zermelo_solution(C, M)

                    if np.isnan(y_pred).any():
                        raise ValueError()
                except:
                    break_count += 1
                    if break_count > 20*S: 
                        raise ValueError('maximum attempts of increasing M passed')
                        return None
                    continue
                    
                if not np.isnan(y_pred).any():
                    spearman = stats.spearmanr(y_pred, ctx_labels)[0]                
                    pearson = stats.pearsonr(y_pred, ctx_labels)[0]                
                    if not np.isnan(spearman):
                        spearman_list.append(spearman)
                        pearson_list.append(pearson)
                    R_count+=1
        
        data = (spearman_list, pearson_list)
        
        if metric == 'debug':
            print(np.mean(spearman_list))
        else:
            save_pickle(data, cache_path)

    else:
        spearman_list, pearson_list = load_pickle(cache_path)

    return spearman_list, pearson_list

#==== Mask Utils ==================================================================================#
def generate_mask(mask:str, R:int, N:int):
    if mask == 'random':  
        M = generate_random_mask(N, R)
    elif mask == 'random-symmetric':  
        M = generate_symmetric_random_mask(N, R)
    elif mask == 'optimal': 
        M = generate_optimal_mask(N, R)
        M = random_order_mask(M)
    elif mask == 'optimal-symmetric': 
        M = generate_optimal_symmetric_mask(N, R)
        M = random_order_mask(M)
    return M

#==== Pickle Utils ================================================================================#
def save_pickle(data, path:str):
    with open(path, 'wb') as output:
        pickle.dump(data, output)

def load_pickle(path:str):
    with open(path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

