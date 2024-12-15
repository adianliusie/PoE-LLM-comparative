import numpy as np
import copy

def merge_sort_indices(N, params, get_probA):
    indices = list(range(N))
    indices = merge_sort(indices, 0, len(indices), input, params, get_probA)
    
    sorted_indices = range(N)
    equal_spaced_scores = [(sorted_indices.index(x)/(N-1)) for x in indices]
        
    num_calls = len(params['comparisons'])
    return equal_spaced_scores, num_calls

def merge_sort(indices, left, right, input, params, get_probA):
    if right - left > 1:
        mid = (left + right) // 2
        merge_sort(indices, left, mid, input, params, get_probA)
        merge_sort(indices, mid, right, input, params, get_probA)
        if params['confidence_beam']:
            merge_with_confidence_beam(indices, left, mid, right, input, params, get_probA)
        else:
            print("We don't support greedy")
    return indices

def get_likelihood_coefficient(N, p):
    x = [0, (N-1)/2, N-1]
    y = [1, 1, 1]

    coefficients = np.polyfit(x, y, 2)  # Fit a 3rd-degree polynomial curve
    func = np.poly1d(coefficients)
    return func(p)

def moving_average(sum,val,idx):
    return sum*idx/(idx+1) + val/(idx+1)

class BeamItem:
    def __init__(self, index_pathway=[], cum_prob=1, pointer_A=-1, pointer_B=-1):
        self.index_pathway = index_pathway
        self.cum_prob = cum_prob
        self.pointer_A = pointer_A
        self.pointer_B = pointer_B

    def __str__(self):
        return f'index_pathway: {self.index_pathway}, cum_prob: {self.cum_prob}'


def merge_with_confidence_beam(indices, left, mid, right, input, params, get_probA):
    # def get_probA(i, j):
    #     if prob_A_matrix[i, j] == 0:
    #         prob_A_matrix[i,j] = C[left_copy[i], right_copy[j]]
    #         params['api_call'] += 1
    #     return prob_A_matrix[i, j]#, uncertainty_matrix[i,j]

    left_copy = indices[left:mid]
    right_copy = indices[mid:right]

    beam_size = params['beam_size']
    prob_A_matrix = np.zeros((len(left_copy), len(right_copy)))  # prob_A_matrix[i, j] is the probability of A better than B 
    # uncertainty_matrix = np.ones_like(prob_A_matrix)  # uncertainty_matrix[i, j] is the uncertainty of A is better
    prob_A_matrix[0, 0] = get_probA(0, 0, params)
    prob_gap = params['prob_gap']

    coef = get_likelihood_coefficient(right-left, 0)
    if prob_A_matrix[0, 0] > 0.5+prob_gap:
        beam = [
            BeamItem(index_pathway=[('B', 0)], cum_prob=np.log(prob_A_matrix[0, 0]+1e-9)*coef, pointer_B=0),
        ]
    elif prob_A_matrix[0, 0] < 0.5-prob_gap:
        beam = [
            BeamItem(index_pathway=[('A', 0)], cum_prob=np.log(1-prob_A_matrix[0, 0]+1e-9)*coef, pointer_A=0),
        ]
    else:
        beam = [
            BeamItem(index_pathway=[('B', 0)], cum_prob=np.log(prob_A_matrix[0, 0]+1e-9)*coef, pointer_B=0),
            BeamItem(index_pathway=[('A', 0)], cum_prob=np.log(1-prob_A_matrix[0, 0]+1e-9)*coef, pointer_A=0),
        ]

    for i in range(len(left_copy)+len(right_copy)-1):
        coef = np.round(get_likelihood_coefficient(right-left, i+1),5)
        # print(coef)
        new_beam = []
        for beam_item in beam:
            for choice in ['A', 'B']:
                beam_item_copy = copy.deepcopy(beam_item)
                if (beam_item_copy.pointer_A < len(left_copy)-1 and beam_item_copy.pointer_B < len(right_copy)-1) \
                    and not (i==len(left_copy)+len(right_copy)-2):
                    prob_A = get_probA(
                            min(beam_item_copy.pointer_A+1, len(left_copy)-1),
                            min(beam_item_copy.pointer_B+1, len(right_copy)-1), 
                            params
                        )
                    if (choice == 'A' and prob_A>0.5+prob_gap) or (choice == 'B' and 1-prob_A > 0.5+prob_gap):
                        continue
                    # beam_item_copy.cum_prob *= 1-prob_A if choice == 'A' else prob_A
                    logprob = np.log(1-prob_A+1e-9) if choice == 'A' else np.log(prob_A+1e-9)
                    beam_item_copy.cum_prob = moving_average(beam_item_copy.cum_prob, logprob*coef, i)

                beam_item_copy.pointer_A += 1 if choice == 'A' else 0
                beam_item_copy.pointer_B += 1 if choice == 'B' else 0

                if (beam_item_copy.pointer_A >= len(left_copy)) or \
                    (beam_item_copy.pointer_B >= len(right_copy)):
                    continue

                current_step = (choice, beam_item_copy.pointer_A if choice == 'A' else beam_item_copy.pointer_B)
                beam_item_copy.index_pathway.append(current_step)
                new_beam.append(beam_item_copy)

        # reduce beam
        new_beam.sort(key=lambda x: x.cum_prob, reverse=True)
        beam = new_beam[:beam_size]

    best_candidate = beam[0]
    sorted_index = []
    for item in best_candidate.index_pathway:
        if item[0] == 'A':
            sorted_index.append(left_copy[item[1]])
        else:
            sorted_index.append(right_copy[item[1]])
            
    indices[left:right] = sorted_index
