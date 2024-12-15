import numpy as np
import scipy as sp
import random
import os

from typing import Tuple
from functools import lru_cache
from scipy import stats
from collections import defaultdict
from tqdm import tqdm

from src.evaluation.loader import SystemLoader
from src.evaluation.eval_tools import Evaluater
from src.utils.general import load_pickle, save_pickle

#== Loading Utils ==============================================================================================================#
def load_comparisons(dataset:str, system:str, score_type:str, balanced:bool=False)->Tuple[np.ndarray, np.ndarray]:
    BASE_PATH = 'define_base_path'
    base_path = f"{BASE_PATH}/output_text"
    path = f"{base_path}/{dataset}/{system}/{score_type}/comparative/outputs/combined.json"

    system = SystemLoader()
    system.load_comparisons_probs(path)
    
    comparisons = system.comparisons_probs
    
    if balanced:
        comparisons = (comparisons + (1-np.transpose(comparisons, (0,2,1))))/2

    ratings_labels = Evaluater.load_ratings_labels(dataset=dataset, score_type=score_type)

    return comparisons, ratings_labels

#== Comparative Masks Utils ====================================================================================================#
def generate_symmetric_random_mask(N:int, num_calls:int=0):
    assert num_calls <= N*(N-1), "number of calls cannot be greater than Nx(N-1)"
    assert num_calls%2==0,       "need an even number of calls for symmetric"

    # randomly select positions from possible matrix (upper area as no repeats)
    possible_indices = [(i, j) for i, j in np.ndindex(N,N) if j>i]
    rand_indices = np.random.choice(range(len(possible_indices)), int(num_calls/2), replace=False)
    selected = [possible_indices[i] for i in rand_indices]
    
    # Assign 1 to the randomly selected indices
    M = np.zeros((N, N), dtype=int)
    for i, j in selected:
        M[i, j] = 1
    
    M_out = M.T + M
    if min(M_out.sum(0)) == 0:
        return generate_symmetric_random_mask(N, num_calls)
    
    return M_out

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
    
    return M

def generate_random_mask(N:int, num_calls:int=0):
    assert num_calls <= N*(N-1), "number of calls cannot be greater than Nx(N-1)"
        
    # duplicate the mask, and take upper triangular since symmetric
    M = np.zeros((N, N), dtype=int)

    num_sel = 0
    for row in np.unique(rows):
        valid_cols = cols[rows == row]  # Columns where the row has '1'
        selected_col = np.random.choice(valid_cols)  # Randomly select one
        
        M_out[row, selected_col] = 1
        num_sel += 1
        
    # randomly select positions from possible matrix
    possible_indices = [(i, j) for i, j in np.ndindex(N,N) if i!=j and M[i,j] == 0]
    rand_indices = np.random.choice(range(len(possible_indices)), num_calls-num_sel, replace=False)
    selected = [possible_indices[i] for i in rand_indices]
    
    # Assign 1 to the randomly selected indices
    for i, j in selected:
        M[i, j] = 1

    if M.sum(axis=0).min() == 0 or M.sum(axis=-1).min() == 0:
        return generate_random_mask(N, num_calls)
    
    return M

#== Generating the Optimal mask (by POE theory) ==============================#
def generate_optimal_symmetric_mask(N, num_calls:int=0):
    assert num_calls <= N*(N-1), "number of calls cannot be greater than Nx(N-1)"
    assert num_calls%2==0,       "need an even number of calls for symmetric"

    num_comparisons = int(num_calls/2)
    W, _ = get_optimal_comparisons(N, R=num_comparisons)
    M = convert_W_to_M(W)
    return M + M.T

def generate_optimal_mask(N, num_calls:int=0):
    assert num_calls <= N*(N-1), "number of calls cannot be greater than Nx(N-1)"    
    W, _ = get_optimal_comparisons(N, R=num_calls)
    M = convert_W_to_M(W)
    return M

@lru_cache(maxsize=5000)
def get_optimal_comparisons(N:int, R:int):
    if R == N-1:
        W, WtW_inv = make_starting_matrix(N)
        return W, WtW_inv
    
    elif R >= N:
        W, WtW_inv = get_optimal_comparisons(N, R-1)

        (i, j) = get_next_comparison(WtW_inv)
        v = np.zeros(N)
        v[i], v[j] = 1, -1

        W = np.concatenate((W, v[None, :]), axis=0)
        WtW_inv = update_WtW_inv(WtW_inv, v[:, None])
        return W, WtW_inv

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

def random_order_mask(M):
    N = M.shape[0]
    order = np.random.permutation(N)
    return M[order][:, order]

#== Comparisons to Ranks Utils =================================================================================================#
def win_ratio(C_tensor, M_tensor):  
    C_tensor_int = (C_tensor > 0.5).astype(int)
    wins = (M_tensor*C_tensor_int).sum(axis=-1) + (M_tensor*(1-C_tensor_int)).sum(axis=-2)
    games = M_tensor.sum(axis=-1) + M_tensor.sum(axis=-2)
    win_ratio = wins/games
    return win_ratio

def avg_prob(C_tensor, M_tensor):  
    wins = (M_tensor*C_tensor).sum(axis=-1) + (M_tensor*(1-C_tensor)).sum(axis=-2)
    games = M_tensor.sum(axis=-1) + M_tensor.sum(axis=-2)
    win_ratio = wins/games
    return win_ratio

def get_MLE_solution_gaussian(C, M:np.ndarray=None, debias=False):
    N = len(C)
    
    if M is None: M = np.ones_like(C) - np.eye(N)
    row_indices, col_indices = np.where(M.astype(bool)) #all indices where M == 1

    K = len(row_indices) + 1
    W = np.zeros((K, N))
    mu = np.zeros(K)
    
    # Set initial conditions
    W[0, 0] = 1
    mu[0] = 0
    
    if debias:
        mean_prob = np.mean(C[M==1])
    else:
        mean_prob = 0.5

    for k, (i, j) in enumerate(zip(row_indices, col_indices), start=1):
        W[k, i] = 1
        W[k, j] = -1
        mu[k] = C[i, j] - mean_prob

    WtW = W.T @ W
    WtW_inv = np.linalg.inv(WtW)
    mle_solution = WtW_inv @ W.T @ mu
    return mle_solution

class LaplaceMinimizer(object):
    def __init__(self, comparison_matrix, sampled_probs):
        self.comparison_matrix = comparison_matrix
        self.sampled_probs = sampled_probs

    def score(self, scores):
        diff = self.comparison_matrix @ scores - self.sampled_probs
        diff = np.mean(np.abs(diff))
        return diff
    
    def gradient(self, scores):
        grad = np.sign(self.comparison_matrix @ scores - self.sampled_probs)
        grad = self.comparison_matrix.T @ grad
        return grad / self.comparison_matrix.shape[0]
        
def get_MLE_solution_laplace(C, M:np.ndarray=None, prior=None):
    N = len(C)
    
    if M is None: M = np.ones_like(C) - np.eye(N)
    row_indices, col_indices = np.where(M.astype(bool)) #all indices where M == 1

    K = len(row_indices) + 1
    W = np.zeros((K, N))
    mu = np.zeros(K)
    
    # Set initial conditions
    W[0, 0] = 1
    mu[0] = 0
    
    mean_prob = np.mean(C[M==1])

    for k, (i, j) in enumerate(zip(row_indices, col_indices), start=1):
        W[k, i] = 1
        W[k, j] = -1
        mu[k] = C[i, j] - mean_prob

    if prior is None:
        prior = np.zeros(N)

    laplace_minimizer = LaplaceMinimizer(W, mu)
    laplace_solution = sp.optimize.fmin_bfgs(
        laplace_minimizer.score, 
        fprime=laplace_minimizer.gradient, 
        x0=prior, 
        disp = False
    )
    return laplace_solution

sigmoid = lambda x: .5 * (np.tanh(.5 * x) + 1)
class SigmoidalGradientMinimizer(object):
    def __init__(self, comparison_matrix, sampled_probs):
        self.comparison_matrix = comparison_matrix
        self.sampled_probs = sampled_probs
        
    def score(self, scores):
        diff = sigmoid(self.comparison_matrix @ scores)
        likelihoods = self.sampled_probs * np.log(diff) + (1 - self.sampled_probs) * np.log(1 - diff)
        likelihood = np.mean(likelihoods)
        return -likelihood
    
    def gradient(self, scores):
        grad = sigmoid(self.comparison_matrix @ scores) - self.sampled_probs
        grad = self.comparison_matrix.T @ grad
        return grad / self.comparison_matrix.shape[0]

class DebiasedSigmoidalGradientMinimizer(object):
    def __init__(self, comparison_matrix, sampled_probs, meanp):
        self.comparison_matrix = comparison_matrix
        self.sampled_probs = sampled_probs
        self.bias = np.log((1 - meanp) / meanp)

    def score(self, scores):
        diff = sigmoid(self.comparison_matrix @ scores - self.bias)
        likelihoods = self.sampled_probs * np.log(diff) + (1 - self.sampled_probs) * np.log(1 - diff)
        likelihood = np.mean(likelihoods)
        return -likelihood
    
    def gradient(self, scores):
        grad = sigmoid(self.comparison_matrix @ scores - self.bias) - self.sampled_probs
        grad = self.comparison_matrix.T @ grad
        return grad / self.comparison_matrix.shape[0]

    
def get_MLE_solution_sigmoid(C, M:np.ndarray=None, debias=False, prior=None):
    N = len(C)
    
    if M is None: M = np.ones_like(C) - np.eye(N)
    row_indices, col_indices = np.where(M.astype(bool)) #all indices where M == 1

    K = len(row_indices)
    W = np.zeros((K, N))
    mu = np.zeros(K)
    
    for k, (i, j) in enumerate(zip(row_indices, col_indices)):
        W[k, i] = 1
        W[k, j] = -1
        mu[k] = C[i, j]

    if prior is None:
        prior = np.zeros(N)

    if debias:
        mean_prob = np.mean(C[M==1])
        sigmoid_minimizer = DebiasedSigmoidalGradientMinimizer(W, mu, mean_prob)
    else:
        sigmoid_minimizer = SigmoidalGradientMinimizer(W, mu)

    sigmoid_solution = sp.optimize.fmin_bfgs(
        sigmoid_minimizer.score, 
        fprime=sigmoid_minimizer.gradient, 
        x0=prior, 
        disp=False
    )
    return sigmoid_solution

class DebiasedSigmoidalGradientMinimizer2(object):
    def __init__(self, comparison_matrix, sampled_probs, gamma):
        self.comparison_matrix = comparison_matrix
        self.sampled_probs = sampled_probs
        self.bias = gamma

    def score(self, scores):
        diff = sigmoid(self.comparison_matrix @ scores - self.bias)
        likelihoods = self.sampled_probs * np.log(diff) + (1 - self.sampled_probs) * np.log(1 - diff)
        likelihood = np.mean(likelihoods)
        return -likelihood
    
    def gradient(self, scores):
        grad = sigmoid(self.comparison_matrix @ scores - self.bias) - self.sampled_probs
        grad = self.comparison_matrix.T @ grad
        return grad / self.comparison_matrix.shape[0]

    
def get_MLE_solution_sigmoid_2(C, M:np.ndarray=None, debias=False, prior=None):
    N = len(C)
    
    if M is None: M = np.ones_like(C) - np.eye(N)
    row_indices, col_indices = np.where(M.astype(bool)) #all indices where M == 1

    K = len(row_indices)
    W = np.zeros((K, N))
    mu = np.zeros(K)
    
    for k, (i, j) in enumerate(zip(row_indices, col_indices)):
        W[k, i] = 1
        W[k, j] = -1
        mu[k] = C[i, j]

    if prior is None:
        prior = np.zeros(N)

    if debias:
        probs = C[M==1]
        gamma = np.pi * np.mean(1/np.tan(np.pi*probs))
        print(1/np.tan(np.pi*probs))

        import time; time.sleep(5)
        meanp = np.mean(probs)
        temp = np.log((1-meanp) / meanp)
        print(gamma)
        print(temp)
        import time; time.sleep(5)
        sigmoid_minimizer = DebiasedSigmoidalGradientMinimizer2(W, mu, gamma)
    else:
        sigmoid_minimizer = SigmoidalGradientMinimizer(W, mu)

    sigmoid_solution = sp.optimize.fmin_bfgs(
        sigmoid_minimizer.score, 
        fprime=sigmoid_minimizer.gradient, 
        x0=prior, 
        disp=False
    )
    return sigmoid_solution

def get_zermelo_solution(C, M=None, max_iter=1000, tol=1e-3):
    assert np.all((C == 0) | (C == 1)) 
    N = len(C)

    if M is None: M = np.ones_like(C) - np.eye(N)
    
    # Initial solution
    ratings = np.exp(np.ones(N))

    # Compute the win matrix
    W = M*C + (M*(1-C)).T
    
    # Add a soft prior to the matrix
    W = W + (1 - np.eye(N)) / (N - 1)
        
    for _ in range(max_iter):
        old_ratings = np.copy(ratings)
                
        # The second denominator term
        scores_matrix = old_ratings[None, :] + old_ratings[:, None]

        # The full denominator matrix
        denom = (W + W.T) / scores_matrix

        # Summation over j
        ratings = W.sum(1) / denom.sum(1)
                
        # Normalization
        ratings = ratings / np.exp(np.log(ratings).mean())

        if np.max(np.abs(ratings - old_ratings)) < tol:
            break
                
    return np.log(ratings)

def get_soft_zermelo_solution(C, M=None, debias=False, max_iter=1000, tol=1e-3):
    N = len(C)

    if M is None: M = np.ones_like(C) - np.eye(N)
    
    # Initial solution
    ratings = np.exp(np.ones(N))

    # Compute the win matrix
    W = M*C + (M*(1-C)).T
    
    # Add a soft prior to the matrix
    # W = W + (1 - np.eye(N)) / (N - 1)
        
    for _ in range(max_iter):
        old_ratings = np.copy(ratings)
                
        # The second denominator term
        scores_matrix = old_ratings[None, :] + old_ratings[:, None]

        # The full denominator matrix
        denom = (W + W.T) / scores_matrix

        # Summation over j
        ratings = W.sum(1) / denom.sum(1)
                
        # Normalization
        ratings = ratings / np.exp(np.log(ratings).mean())

        if np.max(np.abs(ratings - old_ratings)) < tol:
            break
                
    return np.log(ratings)

#==== Analysis Utils ===========================================================================================================#
def get_all_probs_dists(C_list:np.ndarray, labels:np.ndarray, M_list:np.ndarray=None):
    probs_list = []
    dists_list = []

    K, N, _ = C_list.shape
    
    if M_list is None:
        M_list = np.ones_like(C_list) - np.eye(C_list.shape[-1])[None, :]
        
    for k in range(K):
        for i in range(N):
            for j in range(N):
                if M_list[k, i, j] == 0: continue

                prob = C_list[k, i, j]
                diff = labels[k, i] - labels[k, j]

                probs_list.append(prob)
                dists_list.append(diff)
    
    return probs_list, dists_list

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
