import random
import numpy as np
from functools import lru_cache

import sys
import time
import datetime

def get_comparison_ids(N:int, num_comparisons:int, selection_method:str, seed=1):
    np.random.seed(seed)
    random.seed(seed)

    if (selection_method == 'random') or  (selection_method == 'random-doubled'):    
        M, indices = generate_random_mask(N, num_comparisons)
    
    elif selection_method == 'optimal': 
        M, indices = generate_optimal_mask(N, num_comparisons)
        
    ids = []
    for i, j in indices:
        ids.append(f"0-{i}-{j}")
        
        if (selection_method == 'random-doubled'):
            ids.append(f"0-{j}-{i}")
    
    if (selection_method == 'random-doubled'):
        # remove duplicates
        temp_set = set()
        ids = [x for x in ids if not (x in temp_set or temp_set.add(x))]
        M = M + M.T
        M[M==2] = 1
        
    return M, ids

def generate_random_mask(N:int, num_calls:int=0):
    assert num_calls <= N*(N-1), "number of calls cannot be greater than Nx(N-1)"

    # randomly select positions from possible matrix
    possible_indices = [(i, j) for i, j in np.ndindex(N,N) if i!=j]
    rand_indices = np.random.choice(range(len(possible_indices)), num_calls, replace=False)
    selected = [possible_indices[i] for i in rand_indices]
    
    # Assign 1 to the randomly selected indices
    M = np.zeros((N, N), dtype=int)
    for i, j in selected:
        M[i, j] = 1

    if M.sum(axis=0).min() == 0 or M.sum(axis=-1).min() == 0:
        return generate_random_mask(N, num_calls)    
    
    indices = [(i, j) for i, j in np.ndindex(N,N) if M[i,j] == 1]
    random.shuffle(indices)
    return M, indices

def generate_optimal_mask(N, num_calls:int=0):
    assert num_calls <= N*(N-1), "number of calls cannot be greater than Nx(N-1)"    
    sys.setrecursionlimit(200_000)

    W, _, indices = get_optimal_comparisons(N, R=num_calls)
    M = convert_W_to_M(W)
    return M, indices

@lru_cache(maxsize=500000)
def get_optimal_comparisons(N:int, R:int):
    if R == N-1:
        W, WtW_inv = make_starting_matrix(N)
        indices = [(i, i+1) for i in range(N-1)]
        return W, WtW_inv, indices
    
    elif R >= N:
        W, WtW_inv, indices = get_optimal_comparisons(N, R-1)
        
        
        if (R%1000==0): 
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            print(f"{current_time}: optimal selection calculation {R}")
        (i, j) = get_next_comparison(WtW_inv)
        v = np.zeros(N)
        v[i], v[j] = 1, -1

        W = np.concatenate((W, v[None, :]), axis=0)
        WtW_inv = update_WtW_inv(WtW_inv, v[:, None])
        new_indices = indices + [(i, j)]
        return W, WtW_inv, new_indices

def convert_W_to_M(W):
    indices_of_1 = np.where(W[1:] == 1)[1]
    indices_of_neg_1 = np.where(W[1:] == -1)[1]
    
    comparisons = list(zip(indices_of_1, indices_of_neg_1))
    N = max(max(indices_of_1), max(indices_of_neg_1)) + 1
    M = np.zeros((N, N))
    for i, j in comparisons:
        assert M[i, j] + M[j, i] < 2, 'messed up code?'
            
        # randomly determine whether (i, j) or (j, i) is selected (theory is order invariant)
        if M[i, j] == 1:
            M[j, i] = 1
        elif M[j, i] == 1:
            M[i, j] = 1
        elif random.randint(0, 1): 
            M[i, j] = 1
        else:
            M[j, i] = 1

    return M
    
def make_starting_matrix(N:int=9):
    W_start = np.eye(N, k=-1) - np.eye(N)
    W_start[0,0] = 1

    row = np.arange(1.0, N+1)
    column = np.arange(1.0, N+1).reshape(N, 1)
    WtW_inv_start = np.minimum(column, row)
    return W_start, WtW_inv_start

def get_next_comparison(A:np.ndarray):
    # A = WtW_inv
    N = A.shape[0]
    diag = np.diag(A)
    Ai_plus_Aj = diag[:, None] + diag[None, :]
    result = Ai_plus_Aj - 2*A

    max_index = np.unravel_index(np.argmax(result, axis=None), result.shape)    
    return max_index

def update_WtW_inv(WtW_inv:np.ndarray, v:np.ndarray):
    WtW_inv_v = WtW_inv @ v
    num = WtW_inv_v @ WtW_inv_v.T
    den = 1 + v.T @ WtW_inv_v
    WtW_inv_next = WtW_inv - num/den
    return WtW_inv_next
