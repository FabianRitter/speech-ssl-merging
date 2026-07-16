import torch
import torch.nn.functional as F
import scipy
from scipy.optimize import linear_sum_assignment
import sys
import numpy as np
try:
    import ipdb as pdb
except ImportError:
    import pdb
from collections import defaultdict
import networkx as nx

#####################################################################################################################################
############################################################## HELPERS ##############################################################
#####################################################################################################################################

def remove_col(x, idx):
    return torch.cat([x[:, :idx], x[:, idx+1:]], dim=-1)

def compute_correlation(covariance, eps=1e-7):
    covariance = torch.nan_to_num(covariance)
    std = torch.diagonal(covariance).sqrt() # there can be some infs in the covariance matrix
    covariance = covariance / (torch.clamp(torch.nan_to_num(torch.outer(std, std)),min=eps))
    return covariance

def add_bias_to_mats(mats):
    """ Maybe add bias to input. """
    pad_value = 0
    pad_func = torch.nn.ConstantPad1d((0, 1, 0, 1), pad_value)
    biased_mats = []
    for mat in mats:
        padded_mat = pad_func(mat)
        padded_mat[-1, -1] = 1
        biased_mats.append(padded_mat)
    return biased_mats

#####################################################################################################################################
#################################################### MATCHING/ALIGNMENT FUNCTIONS ###################################################
#####################################################################################################################################
def match_tensors_zipit(
    metric, r=.5, a=0.3, b=.125, 
    print_merges=False, get_merge_value=False, add_bias=False, 
    **kwargs
):
    """
    ZipIt! matching algorithm. Given metric dict, computes matching as defined in paper. 
    Args:
    - metric: dictionary containing metrics. This must contain either a covariance or cossim matrix, and 
        must be [(num_models x model_feature_dim), (num_models x model_feature_dim)]. 
    - r: Amount to reduce total input feature dimension - this is num_models x model_feature_dim. This function will
        compute (un)merge matrix that goes from 
        (num_models x model_feature_dim) -> (1-r)*(num_models x model_feature_dim) = merged_feature_dim.
        E.g. if num_models=2, model_feature_dim=10 and r=.5, the matrix will map from 2x10=20 -> (1-.5)x2x10=10, or halve the 
        collective feature space of the models.
    - a: alpha hyperparameter as defined in Section 4.3 of our paper. 
    - b: beta hyperparameter as defined in Section 4.3 of our paper.
    - print_merges: whether to print computed (un)merge matrices.
    - get_merge_value default False, returns the sum of correlations over all the merges which the algorithm made. 
    - add_bias: whether to add a bias to the input. This should only be used if your module expects the input with bias offset.
    returns:
    - (un)merge matrices
    """
    print(f"ZipIt! matching with r={r}, a={a}, b={b}")
    if "covariance" in metric:
        sims = compute_correlation(metric["covariance"])
    elif "cossim" in metric:
        sims = metric["cossim"]
    O = sims.shape[0]
    remainder = int(O * (1-r) + 1e-4)
    permutation_matrix = torch.eye(O, O)#, device=sims.device)

    torch.diagonal(sims)[:] = -torch.inf

    num_models = int(1/(1 - r) + 0.5)
    Om = O // num_models

    original_model = torch.zeros(O, device=sims.device).long()
    for i in range(num_models):
        original_model[i*Om:(i+1)*Om] = i

    to_remove = permutation_matrix.shape[1] - remainder
    budget = torch.zeros(num_models, device=sims.device).long() + int((to_remove // num_models) * b + 1e-4)

    merge_value = []
    sims_copy = sims.clone()
    merge_values_list = [] # Store individual correlation values of merges


    while permutation_matrix.shape[1] > remainder:
        best_idx = sims.reshape(-1).argmax()
        row_idx = best_idx % sims.shape[1]
        col_idx = best_idx // sims.shape[1]
        
        merge_value.append(permutation_matrix[row_idx, col_idx])
        merge_values_list.append(sims_copy[col_idx, row_idx].item()) # Use item() to get float

        if col_idx < row_idx:
            row_idx, col_idx = col_idx, row_idx
        
        row_origin = original_model[row_idx]
        col_origin = original_model[col_idx]
        
        permutation_matrix[:, row_idx] += permutation_matrix[:, col_idx]
        permutation_matrix = remove_col(permutation_matrix, col_idx)
        
        sims[:, row_idx] = torch.minimum(sims[:, row_idx], sims[:, col_idx])
        
        if 'magnitudes' in metric:
            metric['magnitudes'][row_idx] = torch.minimum(metric['magnitudes'][row_idx], metric['magnitudes'][col_idx])
            metric['magnitudes'] = remove_col(metric['magnitudes'][None], col_idx)[0]
        
        if a <= 0:
            sims[row_origin*Om:(row_origin+1)*Om, row_idx] = -torch.inf
            sims[col_origin*Om:(col_origin+1)*Om, row_idx] = -torch.inf
        else: sims[:, row_idx] *= a
        sims = remove_col(sims, col_idx)
        
        sims[row_idx, :] = torch.minimum(sims[row_idx, :], sims[col_idx, :])
        if a <= 0:
            sims[row_idx, row_origin*Om:(row_origin+1)*Om] = -torch.inf
            sims[row_idx, col_origin*Om:(col_origin+1)*Om] = -torch.inf
        else: sims[row_idx, :] *= a
        sims = remove_col(sims.T, col_idx).T

        row_origin, col_origin = original_model[row_idx], original_model[col_idx]
        original_model = remove_col(original_model[None, :], col_idx)[0]
        
        if row_origin == col_origin:
            origin = original_model[row_idx].item()
            budget[origin] -= 1

            if budget[origin] <= 0:
                # kill origin
                selector = original_model == origin
                sims[selector[:, None] & selector[None, :]] = -torch.inf
    
    if add_bias:
        unmerge_mats = permutation_matrix.chunk(num_models, dim=0)
        unmerge_mats = add_bias_to_mats(unmerge_mats)
        unmerge = torch.cat(unmerge_mats, dim=0)
    else:
        unmerge = permutation_matrix

    merge = permutation_matrix / (permutation_matrix.sum(dim=0, keepdim=True) + 1e-5)
    if print_merges:
        O, half_O = unmerge.shape
        A_merge, B_merge = unmerge.chunk(2, dim=0)
        
        A_sums = A_merge.sum(0)
        B_sums = B_merge.sum(0)
        
        A_only = (B_sums == 0).sum()
        B_only = (A_sums == 0).sum()
        
        overlaps = half_O - (A_only + B_only)
        
        print(f'A into A: {A_only} | B into B: {B_only} | A into B: {overlaps}')
        print(f'Average Connections: {unmerge.sum(0).mean()}')
    
    merge = merge.to(sims.device)
    unmerge = unmerge.to(sims.device)
    cost = np.mean(merge_values_list) if merge_values_list else 0.0
    print(f"    Calculated ZipIt Cost (avg correlation of merges): {cost:.4f}") # Debug print
    if get_merge_value:
        merge_value = sum(merge_value) / len(merge_value)
        return merge.T, unmerge, cost, cost

    return merge.T, unmerge, None, cost


def match_tensors_permute_symmetric(r=0.5, get_merge_value=False,
                                    print_costs=True, no_absval=False,
                                    correlation_matrix=None, **kwargs):
    """
    A symmetric merging function.
    It computes two permutation matrices:
      - P: mapping from MERT channels to HuBERT (using the top–right block)
      - Q: mapping from HuBERT channels to MERT (using the bottom–left block)
    Then it combines them by averaging (i.e. symmetric_merge = (P + Q.T) / 2).
    Returns:
      symmetric_merge: merge matrix
      symmetric_unmerge: approximate inverse (transpose)
      (optionally merge_value and cost)
    """
    if correlation_matrix is None:
        raise ValueError("A correlation matrix is required.")

    correlation = correlation_matrix
    O = correlation.shape[0]
    N = int(1 / (1 - r) + 0.5)
    Om = O // N
    device = correlation.device

    

    # Compute mapping for MERT -> HuBERT (using the top-right block)
    corr_mert_to_hubert = correlation[:Om, Om:2*Om].cpu().numpy()
    if not no_absval:
        corr_mert_to_hubert = np.abs(corr_mert_to_hubert)
    row_ind_p, col_ind_p = linear_sum_assignment(corr_mert_to_hubert, maximize=True)
    # Create a permutation matrix P from the assignment (columns reordered)
    P = torch.eye(Om, device=device)[torch.tensor(col_ind_p).long().to(device)].T
    cost_p = corr_mert_to_hubert[row_ind_p, col_ind_p].sum()

    # Compute mapping for HuBERT -> MERT (using the bottom-left block)
    corr_hubert_to_mert = correlation[Om:2*Om, :Om].cpu().numpy()
    if not no_absval:
        corr_hubert_to_mert = np.abs(corr_hubert_to_mert)
    row_ind_q, col_ind_q = linear_sum_assignment(corr_hubert_to_mert, maximize=True)
    Q = torch.eye(Om, device=device)[torch.tensor(col_ind_q).long().to(device)].T
    cost_q = corr_hubert_to_mert[row_ind_q, col_ind_q].sum()

    # Combine the two permutation matrices symmetrically.
    # Here, we average P (mapping MERT->HuBERT) with the transpose of Q (mapping HuBERT->MERT)
    symmetric_merge = (P + Q.T) / 2
    symmetric_unmerge = symmetric_merge.T  # approximate inverse


    #mats = [torch.eye(Om, device=device), symmetric_merge]
    #merge = torch.cat(mats, dim=1)  # [Om, 2*Om]
    #unmerge = torch.cat([torch.eye(Om, device=device), symmetric_unmerge], dim=0)

    if print_costs:
        print(f"Cost (MERT -> HuBERT): {cost_p / Om:.4f}, Cost (HuBERT -> MERT): {cost_q / Om:.4f}")

    if get_merge_value:
        merge_value = (cost_p + cost_q) / (2 * Om)
        return symmetric_merge, symmetric_unmerge, merge_value
    
    return symmetric_merge, symmetric_unmerge, None, (cost_p + cost_q) / (2 * Om)
    
def match_tensors_intra_pairwise(tensor, reduction_ratio=0.5, no_absval=False,node_id_for_debug=None, **kwargs):
    """
    Given a 2D tensor (e.g. rows represent channels/features), computes pairwise merge.
    
    Args:
      tensor: a torch.Tensor of shape (C, D) (C = number of channels/features).
      reduction_ratio: fraction of channels to retain (e.g., 0.5 will merge in pairs).
      no_absval: if False, uses absolute correlation values.
      
    Returns:
      merge_matrix: Tensor of shape (C, C_new) to project from C to reduced dimension.
      unmerge_matrix: Tensor of shape (C_new, C) to expand back (used for weight re–construction).
      cost: average correlation cost for diagnostic purposes.
    """
    
    
    C, D = tensor.shape
    C_new = int(C * reduction_ratio)
    
    if C % 2 != 0:
        raise ValueError("For pairwise merging, the number of channels must be even.")

    # Compute correlation matrix among channels (using rows of tensor as feature vectors).
    tensor_float = tensor.float()
    tensor_centered = tensor_float - tensor_float.mean(dim=1, keepdim=True)
    cov = torch.matmul(tensor_centered, tensor_centered.T)  # shape: (C, C)

    std = torch.sqrt(torch.diag(cov) + 1e-7)
    corr = cov / (std.unsqueeze(1) * std.unsqueeze(0) + 1e-7)

    problem_nodes = {7, 11, 15, 19} # Nodes we are interested in
    is_problem_node = node_id_for_debug in problem_nodes

    if is_problem_node:
        print(f"\n--- Debugging Node {node_id_for_debug} ---")
        print(f"Input tensor shape: {tensor_centered.shape}")
        print(f"cov diagonal is is {torch.diag(cov)}")
        if tensor_centered.numel() == 0:
             print("ERROR: Input tensor is empty!")

    if not no_absval:
        corr = torch.abs(corr)
    
    # Prevent self–matching by setting diagonal to -inf.
    corr = corr - torch.diag(torch.full((C,), float('inf'), device=tensor_centered.device))
    
    # Build a complete graph where each node is a channel.
    G = nx.Graph()
    G.add_nodes_from(range(C))
    # Add edge (i, j) with weight = correlation value.
    for i in range(C):
        for j in range(i+1, C):
            # Use .item() to convert tensor value to float.
            G.add_edge(i, j, weight=corr[i, j].item())
    
    # Compute maximum weighted matching with full cardinality.
    matching = nx.max_weight_matching(G, maxcardinality=True)
    pairs = [tuple(sorted(edge)) for edge in matching]
    
    if len(pairs) != C_new:
        print("Pairing did not produce the expected number of merged channels.")
        raise ValueError("Pairing did not produce the expected number of merged channels.")
    
    # Build merge (projection) and unmerge matrices.
    merge_matrix = torch.zeros(C, C_new, device=tensor.device)
    unmerge_matrix = torch.zeros(C_new, C, device=tensor.device)
    for idx, (i, j) in enumerate(pairs):
        merge_matrix[i, idx] = 0.5
        merge_matrix[j, idx] = 0.5
        # In this simple example, unmerging simply copies the merged value back.
        unmerge_matrix[idx, i] = 1.0
        unmerge_matrix[idx, j] = 1.0

    cost = np.mean([corr[i, j].item() for i, j in pairs])
    print(f"cost is {cost}")

    return merge_matrix, unmerge_matrix, cost

def match_tensors_intra_pairwise_MHA(n_heads, tensor, reduction_ratio=0.5, no_absval=False,**kwargs):
    """
    Apply intra-pairwise merging per attention head.

    Args:
        n_heads (int): Number of attention heads.
        tensor (torch.Tensor): Shape [n_heads * head_dim, D], where D is samples.
        reduction_ratio (float): Fraction of channels to retain per head.
        no_absval (bool): If True, use raw correlations.

    Returns:
        merge (torch.Tensor): Shape [n_heads * head_dim, n_heads * reduced_head_dim].
        unmerge (torch.Tensor): Shape [n_heads * reduced_head_dim, n_heads * head_dim].
        cost (float): Average cost over heads.
    """
    O, D = tensor.shape
    head_dim = O // n_heads
    if O % n_heads != 0:
        raise ValueError("Tensor dimension must be divisible by n_heads.")

    merge_matrices = []
    unmerge_matrices = []
    total_cost = 0.0
    for h in range(n_heads):
        start = h * head_dim
        end = (h + 1) * head_dim
        head_tensor = tensor[start:end, :]  # [head_dim, D]
        merge_head, unmerge_head, cost = match_tensors_intra_pairwise(
            head_tensor, reduction_ratio=reduction_ratio, no_absval=no_absval
        )
        merge_matrices.append(merge_head)
        unmerge_matrices.append(unmerge_head)
        total_cost += cost

    avg_cost = total_cost / n_heads
    merge = torch.block_diag(*merge_matrices)  # [O, O_new]
    unmerge = torch.block_diag(*unmerge_matrices)  # [O_new, O]
    return merge, unmerge, avg_cost





def match_tensors_permute_v2(r=.5, get_merge_value=False,
                          print_costs=False, no_absval=False,
                          correlation_matrix=None,**kwargs):
    """
    Modified to consider full correlation matrix (HuBERT+MERT vs. HuBERT+MERT).
    Still primarily maps MERT to HuBERT space, but with potential for slight
    intra-model adjustments indirectly.
    """
    correlation = correlation_matrix
    O = correlation.shape[0] # Total dimension (HuBERT + MERT features concatenated)
    N = 2 # Fixed to 2 models for now (HuBERT and MERT)
    Om = O // N          # Dimension of each model's features

    device = correlation.device

    mats = [torch.eye(Om, device=device)] # Identity for HuBERT (graph[0])
    cost = 0

    # Define weights for different correlation blocks (experiment with these)
    weight_inter_model = 1.0   # Weight for HuBERT-MERT and MERT-HuBERT correlations
    weight_intra_model = 0.2   # Weight for HuBERT-HuBERT and MERT-MERT correlations (lower weight)

    # Construct a combined cost matrix for Hungarian algorithm
    combined_cost_matrix = torch.zeros((Om, Om), dtype=torch.float64)

    # 1. HuBERT-MERT correlation (top-right) - Main mapping direction
    corr_matrix_hm = correlation[:Om, Om:2*Om].cpu().numpy()
    if not no_absval:
        corr_matrix_hm = np.absolute(corr_matrix_hm)
    combined_cost_matrix += torch.tensor(corr_matrix_hm) * weight_inter_model

    # 2. MERT-HuBERT correlation (bottom-left) -  Consider reverse direction (optional weighting)
    corr_matrix_mh = correlation[Om:2*Om, :Om].T.cpu().numpy() # Transpose for correct shape
    if not no_absval:
        corr_matrix_mh = np.absolute(corr_matrix_mh)
    combined_cost_matrix += torch.tensor(corr_matrix_mh) * weight_inter_model # You could use a different weight here if needed

    # 3. HuBERT-HuBERT correlation (top-left) - Intra-HuBERT similarity (optional, lower weight)
    corr_matrix_hh = correlation[:Om, :Om].cpu().numpy()
    if not no_absval:
        corr_matrix_hh = np.absolute(corr_matrix_hh)
    combined_cost_matrix += torch.tensor(corr_matrix_hh) * weight_intra_model

    # 4. MERT-MERT correlation (bottom-right) - Intra-MERT similarity (optional, lower weight)
    corr_matrix_mm = correlation[Om:2*Om, Om:2*Om].cpu().numpy()
    if not no_absval:
        corr_matrix_mm = np.absolute(corr_matrix_mm)
    combined_cost_matrix += torch.tensor(corr_matrix_mm) * weight_intra_model


    try:
        row_ind, col_ind = linear_sum_assignment(
            combined_cost_matrix.numpy(), maximize=True) # Hungarian algorithm on combined cost
        cost = combined_cost_matrix.numpy()[row_ind, col_ind].sum()
    except Exception:
        raise

    new_mat = torch.eye(Om, device=device)[torch.tensor(col_ind).long().to(device)]
    mats.append(new_mat.T)

    unmerge_mats = mats
    unmerge = torch.cat(unmerge_mats, dim=0)
    merge = torch.cat(mats, dim=0)
    merge = merge / (merge.sum(dim=0, keepdim=True) + 1e-5)

    if print_costs:
        cost = cost / merge.shape[0]
        print(f'cost (v2): {cost}')

    return merge.T, unmerge, None, cost / merge.shape[0]

def match_tensors_permute(r=.5, get_merge_value=False, 
                          print_costs=False, no_absval=False,
                          correlation_matrix=None,**kwargs):
    """
    This function is adapted from ZipIt! (https://github.com/gstoica27/ZipIt)

    Matches arbitrary models by permuting all to the spaces of the first in your graph list. 
    Mimics Rebasin methods. 
    """

    correlation = correlation_matrix

    O = correlation.shape[0]
    N = int(1/(1 - r) + 0.5)
    Om = O // N
    device = correlation.device
    
    mats = [torch.eye(Om, device=device)]
    cost = 0

    for i in range(1, N): #only computing for i=1!
        try:
            corr_matrix = correlation[:Om, Om*i:Om*(i+1)].cpu().numpy()
            if no_absval == False:
                corr_matrix = np.absolute(corr_matrix)
            row_ind, col_ind = linear_sum_assignment(
                corr_matrix, maximize=True)
            cost =  corr_matrix[row_ind, col_ind].sum()
            # correlation subset is is [0:4096, 4096:8192]
            # correlation between the first graph's and second graph's features
        except Exception:
            raise

        mats.append(torch.eye(Om, device=device)[torch.tensor(col_ind).long().to(device)].T)

    unmerge_mats = mats
        
    unmerge = torch.cat(unmerge_mats, dim=0)
    merge = torch.cat(mats, dim=0)
    #merge = merge / (merge.sum(dim=0, keepdim=True) + 1e-5)
    if get_merge_value:
        merge_value = correlation[:Om, Om*i:Om*(i+1)].cpu().numpy()[row_ind, col_ind].mean()
        return merge.T, unmerge, merge_value
    if print_costs:
        cost = cost / merge.shape[0]
        print(f'cost: {cost}')
    
    

    return merge.T, unmerge, None, cost / merge.shape[0]

def match_tensors_permute_MHA(n_heads, permute_heads=False, 
                              head_assignments=[], r=.5, get_merge_value=False, 
                              print_costs=True, no_absval=False, 
                              correlation_matrix=None, **kwargs):
    """
    Handles different head permutations in attention
    """
    correlation = correlation_matrix

    O = correlation.shape[0]

    N = int(1/(1 - r) + 0.5) # num models
    Om = O // N # matrix dimension
    device = correlation.device
    query_size = Om // n_heads 
    
    mats = [torch.eye(Om, device=device)]
    head_perms = []

    # compute head perms in order
    if permute_heads == False:
        cost = 0
        for i in range(1, N): #just once if 2 models]
            for j in range(n_heads):
                try:
                    # by head
                    corr_submatrix =  correlation[query_size * j:query_size * (j+1), Om*i + query_size*j:Om*i + query_size*(j+1)].cpu().numpy()
                    if no_absval == False:
                        corr_submatrix = np.absolute(corr_submatrix)
                    row_ind, col_ind = linear_sum_assignment(corr_submatrix, maximize=True)


                    head_perms.append(torch.tensor(col_ind + j*query_size))
                    cost += corr_submatrix[row_ind, col_ind].sum()
                    
                    # for whole model correlation subset is is [0:4096, 4096:8192]
                    # correlation between the first graph's and second graph's features
                except Exception:
                    raise
        outer_col_ind = np.arange(n_heads)
    # compute head perms out of order according to predefined ordering or find our own
    elif permute_heads == True:
        cost = 0
        col_inds_storage = defaultdict(lambda: defaultdict(int))
        if head_assignments != []:
            outer_row_ind = np.arange(n_heads)
            outer_col_ind = head_assignments
            for i in range(n_heads):
                head1_idx = [query_size * outer_row_ind[i], query_size * (outer_row_ind[i] + 1)]
                head2_idx = [Om + query_size * outer_col_ind[i], Om + query_size * (outer_col_ind[i] + 1)]
                # take abs value of submatrix of correlations
                corr_submatrix = correlation[head1_idx[0]:head1_idx[1], head2_idx[0]:head2_idx[1]].cpu().numpy()
                if no_absval == False:
                    corr_submatrix = np.absolute(corr_submatrix)
                # compute perm for head j & head k 
                row_ind, col_ind = linear_sum_assignment(corr_submatrix, maximize=True)

                cost += corr_submatrix[row_ind, col_ind].sum()
                col_inds_storage[outer_row_ind[i]][outer_col_ind[i]] = col_ind
           
        else: 
            costs = np.ones((n_heads, n_heads)) * -sys.maxsize  # cost matrix for hungarian algo steps
            for i in range(1, N):  #just once if 2 models 
                for j in range(n_heads): # outer loop through all heads
                    for k in range(n_heads):  # inner loop through heads >= current head j
                        head1_idx = [query_size * j, query_size * (j+1)]
                        head2_idx = [Om * i + query_size * k, Om * i + query_size * (k+1)]

                        # take abs value of submatrix of correlations
                        corr_submatrix = correlation[head1_idx[0]:head1_idx[1], head2_idx[0]:head2_idx[1]].cpu().numpy()
                        if no_absval == False:
                            corr_submatrix = np.absolute(corr_submatrix)
                        # compute perm for head j & head k 
                        row_ind, col_ind = linear_sum_assignment(corr_submatrix, maximize=True)

                        # store cost (cost is maximized here)
                        costs[j,k] = corr_submatrix[row_ind, col_ind].sum()
                        #costs[k,j] = costs[j,k] # make symmetric

                        # store perm so we don't have to recompute it later
                        col_inds_storage[j][k] = col_ind

            outer_row_ind, outer_col_ind = linear_sum_assignment(costs, maximize=True) # get assignment with lowest cost
            cost += costs[outer_row_ind, outer_col_ind].sum()

        for j in range(n_heads):
            head_1 = outer_row_ind[j] # these are in order, outer_row_ind[j] = j
            head_2 = outer_col_ind[j]

            head_perm = col_inds_storage[head_1][head_2]
            head_perms.append(torch.tensor(head_perm + query_size*head_2))

    new_mat = torch.eye(Om, device=device)[torch.tensor(torch.cat(head_perms)).long().to(device)]
    mats.append(new_mat.T)
    
    unmerge_mats = mats
    
    unmerge = torch.cat(unmerge_mats, dim=0)
    merge = torch.cat(mats, dim=0)
    #merge = merge / (merge.sum(dim=0, keepdim=True) + 1e-5)
    if print_costs:
        cost = cost / merge.shape[0]
        print(f'cost: {cost}')
    if get_merge_value:
        merge_value = correlation[:Om, Om*i:Om*(i+1)].cpu().numpy()[row_ind, col_ind].mean()
        return merge.T, unmerge, merge_value
    return merge.T, unmerge, outer_col_ind, cost / merge.shape[0]

def match_tensors_zipit_MHA(n_heads, metric, r=0.5, a=0.3, b=0.125, print_merges=False, **kwargs):
    """
    ZipIt! merging applied per attention head.
    Args:
        n_heads (int): Number of attention heads.
        metric (dict): Contains 'covariance' matrix of shape [TotalDim, TotalDim].
    Returns:
        merge.T (torch.Tensor): [FinalDim, TotalOriginalDim]
        unmerge (torch.Tensor): [TotalOriginalDim, FinalDim]
    """
    covariance = metric['covariance']
    total_dim = covariance.shape[0]
    num_models = 2  # Assuming HuBERT + MERT
    dim_per_model = total_dim // num_models
    head_dim = dim_per_model // n_heads

    merge_mats = []
    unmerge_mats = []

    for h in range(n_heads):
        start = h * head_dim
        end = (h + 1) * head_dim
        # Extract head-specific covariance for both models
        head_indices = list(range(start, end)) + list(range(dim_per_model + start, dim_per_model + end))
        head_cov = covariance[head_indices][:, head_indices]
        head_metric = {'covariance': head_cov}
        # Apply ZipIt! per head
        head_merge_T, head_unmerge = match_tensors_zipit(head_metric, r=r, a=a, b=b, print_merges=print_merges, **kwargs)
        merge_mats.append(head_merge_T)
        unmerge_mats.append(head_unmerge)

    # Combine head-specific matrices
    merge_T = torch.block_diag(*merge_mats)  # [Sum(FinalHeadDims), TotalOriginalDim]
    unmerge = torch.block_diag(*unmerge_mats)  # [TotalOriginalDim, Sum(FinalHeadDims)]
    return merge_T, unmerge

def match_tensors_identity(r=.5, correlation_matrix=None, **kwargs):
    # weight averaging.  
    
    correlation = correlation_matrix
    O = correlation.shape[0]

    N = int(1/(1 - r) + 0.5)
    Om = O // N
    device = correlation.device
    corr_matrix = correlation[:Om, Om:Om*2].cpu().numpy()
    cost = corr_matrix.trace()

    mats = [torch.eye(Om, device=device) for _ in range(N)]
    
    unmerge_mats = mats

    unmerge = torch.cat(unmerge_mats, dim=0)
    merge = torch.cat(mats, dim=0)
    merge = merge / (merge.sum(dim=0, keepdim=True) + 1e-5)
    cost = cost / merge.shape[0]
    return merge.T, unmerge, None, cost

def match_tensors_permute_symmetric_MHA(n_heads, permute_heads=True, head_assignments=None,
                                          r=0.5, get_merge_value=False, print_costs=False,
                                          no_absval=False, correlation_matrix=None, **kwargs):
    """
    Symmetric matching for multi-head attention that handles:
      1. Intra-head (channel-level) permutation: for each head, compute a symmetric mapping.
      2. Inter-head (head order) permutation: compute an optimal symmetric reordering of heads.
      
    Parameters:
      n_heads (int): number of attention heads.
      permute_heads (bool): if True, perform head-order permutation.
      head_assignments (optional): if provided, a pre-specified assignment for heads.
      r (float): reduction ratio parameter.
      get_merge_value (bool): if True, return a merge cost value.
      print_costs (bool): if True, print computed costs.
      no_absval (bool): if True, do not take absolute value of correlation submatrices.
      correlation_matrix (torch.Tensor): full correlation matrix, assumed shape [O, O] where
           O = 2 * Om and Om is the number of channels per model.
           
    Returns:
      merge (torch.Tensor): final merged matrix (transposed as in original code)
      unmerge (torch.Tensor): corresponding unmerge matrix (approximate inverse)
      extra (depending on get_merge_value, either a merge value or outer assignment)
      cost (float): average cost per head.
    """
    if correlation_matrix is None:
        raise ValueError("A correlation matrix is required.")
        
    correlation = correlation_matrix
    O = correlation.shape[0]
    N = int(1 / (1 - r) + 0.5)  # number of models (for two models, N=2)
    Om = O // N               # number of channels per model
    device = correlation.device
    query_size = Om // n_heads  # number of channels per head

    # ----- Step 1: Compute symmetric intra-head (channel-level) permutations -----
    per_head_merge = []  # list to store symmetric merge matrices for each head
    per_head_cost = []   # list to store average cost for each head
    
    for j in range(n_heads):
        start = query_size * j
        end = query_size * (j + 1)
        # Block from Model A to Model B (top-right block)
        block_A_to_B = correlation[start:end, Om + start:Om + end].cpu().numpy()
        if not no_absval:
            block_A_to_B = np.abs(block_A_to_B)
        row_ind, col_ind = linear_sum_assignment(block_A_to_B, maximize=True)
        P = torch.eye(query_size, device=device)[torch.tensor(col_ind).long().to(device)].T
        cost_A_to_B = block_A_to_B[row_ind, col_ind].sum()
        
        # Block from Model B to Model A (bottom-left block)
        block_B_to_A = correlation[Om + start:Om + end, start:end].cpu().numpy()
        if not no_absval:
            block_B_to_A = np.abs(block_B_to_A)
        row_ind2, col_ind2 = linear_sum_assignment(block_B_to_A, maximize=True)
        Q = torch.eye(query_size, device=device)[torch.tensor(col_ind2).long().to(device)].T
        cost_B_to_A = block_B_to_A[row_ind2, col_ind2].sum()
        
        # Average the two mappings symmetrically
        sym_head_merge = (P + Q.T) / 2.0
        per_head_merge.append(sym_head_merge)
        per_head_cost.append((cost_A_to_B + cost_B_to_A) / 2.0)
        
    avg_intra_cost = np.mean(per_head_cost)

    # ----- Step 2: Compute symmetric head-order permutation (inter-head) if requested -----
    if permute_heads:
        # Build a cost matrix between heads.
        # For head j in Model A and head k in Model B, extract corresponding block
        head_cost_matrix = np.zeros((n_heads, n_heads))
        for j in range(n_heads):
            for k in range(n_heads):
                # For Model A head j, block is rows [j*query_size:(j+1)*query_size]
                # For Model B head k, block is columns [Om + k*query_size:Om + (k+1)*query_size]
                block = correlation[j*query_size:(j+1)*query_size, Om + k*query_size:Om + (k+1)*query_size].cpu().numpy()
                if not no_absval:
                    block = np.abs(block)
                # Cost here can be the average (or sum) of the block's entries.
                head_cost_matrix[j, k] = np.mean(block)
        # Make the cost matrix symmetric
        head_cost_matrix = (head_cost_matrix + head_cost_matrix.T) / 2.0
        # Compute assignment for head reordering using the Hungarian algorithm
        outer_row_ind, outer_col_ind = linear_sum_assignment(head_cost_matrix, maximize=True)
        # Reorder the per-head merge matrices accordingly.
        per_head_merge_reordered = [None] * n_heads
        for idx, j in enumerate(outer_row_ind):
            k = outer_col_ind[idx]
            # For symmetry, average the merge matrix of head j and head k if they differ.
            # In an ideal symmetric case these should be similar.
            merged_head = (per_head_merge[j] + per_head_merge[k]) / 2.0
            per_head_merge_reordered[j] = merged_head
        # If head_assignments were provided, you might alternatively use them.
        outer_assignment = outer_col_ind
        outer_cost = head_cost_matrix[outer_row_ind, outer_col_ind].mean()
    else:
        # If not reordering, use natural order.
        per_head_merge_reordered = per_head_merge
        outer_assignment = np.arange(n_heads)
        outer_cost = 0.0


    # ----- Step 3: Assemble the full merge matrix from re-ordered per-head merge matrices -----
    # Create an empty merge matrix for the attention layer (size Om x Om)
    full_merge = torch.zeros((Om, Om), device=device)
    for j in range(n_heads):
        start = j * query_size
        end = (j + 1) * query_size
        full_merge[start:end, start:end] = per_head_merge_reordered[j]
    
    # Optionally, combine the intra-head and inter-head costs
    total_cost = avg_intra_cost + outer_cost

    # The original code appends an identity block to create a larger matrix.
    mats = [torch.eye(Om, device=device), full_merge.T]
    unmerge = torch.cat(mats, dim=0)
    merge = torch.cat(mats, dim=0)
    merge = merge / (merge.sum(dim=0, keepdim=True) + 1e-5)
    
    avg_cost = total_cost / n_heads  # averaged over heads (adjust as needed)
    if print_costs:
        print(f'Avg intra-head cost: {avg_intra_cost:.4f}, Avg head-order cost: {outer_cost:.4f}, Overall avg cost: {avg_cost:.4f}')
    if get_merge_value:
        merge_value = avg_cost
        return merge.T, unmerge, merge_value
    return merge.T, unmerge, outer_assignment, avg_cost


#####################################################################################################################################
########################################## SINKHORN-CC (G4.1): Differentiable Permutation ###########################################
#####################################################################################################################################

def _sinkhorn_normalization(log_alpha, num_iters=20):
    """Apply Sinkhorn normalization to produce a doubly-stochastic matrix.

    Args:
        log_alpha: [D, D] unnormalized logit matrix
        num_iters: number of row/column normalization iterations
    Returns:
        [D, D] doubly-stochastic matrix
    """
    for _ in range(num_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)


def _sinkhorn_temperature(step, total_steps, tau_max=1.0, tau_min=0.01):
    """Exponential temperature annealing from tau_max to tau_min."""
    if total_steps <= 1:
        return tau_min
    progress = step / (total_steps - 1)
    return tau_max * (tau_min / tau_max) ** progress


def _sinkhorn_optimize(cross_corr, Om, device, no_absval=False,
                       sinkhorn_iters=20, num_opt_steps=300, lr=0.01,
                       tau_max=1.0, tau_min=0.01, seed=42, print_costs=False):
    """Core Sinkhorn optimization loop shared by both regular and MHA variants.

    Args:
        cross_corr: [D, D] cross-correlation block between model A and B
        Om: dimension per model
        device: torch device
    Returns:
        (P, cost) - permutation matrix [D, D] and alignment cost
    """
    torch.manual_seed(seed)

    # Warm-start logits from cross-correlation
    init_logits = cross_corr.abs() if not no_absval else cross_corr.clone()
    log_alpha = init_logits.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([log_alpha], lr=lr)

    # Optimization target: the correlation matrix itself
    target = cross_corr.abs().detach() if not no_absval else cross_corr.detach()

    best_loss = float('inf')
    best_log_alpha = log_alpha.data.clone()

    for step in range(num_opt_steps):
        optimizer.zero_grad()
        tau = _sinkhorn_temperature(step, num_opt_steps, tau_max, tau_min)
        S = _sinkhorn_normalization(log_alpha / tau, num_iters=sinkhorn_iters)

        # Loss: maximize trace(S @ C_AB) = how well S aligns features
        alignment = torch.trace(S @ target)
        loss = -alignment / Om

        loss.backward()
        torch.nn.utils.clip_grad_norm_([log_alpha], max_norm=10.0)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_log_alpha = log_alpha.data.clone()

        if print_costs and step % 50 == 0:
            print(f"    step {step}/{num_opt_steps}, tau={tau:.4f}, loss={loss.item():.6f}")

    # Extract hard permutation via Hungarian on the final soft matrix
    with torch.no_grad():
        S_final = _sinkhorn_normalization(
            best_log_alpha / tau_min, num_iters=sinkhorn_iters * 2
        )

    S_np = S_final.detach().cpu().numpy()
    if not no_absval:
        S_np = np.absolute(S_np)
    row_ind, col_ind = linear_sum_assignment(S_np, maximize=True)

    # Compute cost on the ORIGINAL correlation for fair comparison with G1
    corr_np = cross_corr.cpu().numpy()
    if not no_absval:
        corr_np = np.absolute(corr_np)
    cost = corr_np[row_ind, col_ind].sum()

    P = torch.eye(Om, device=device)[torch.tensor(col_ind).long().to(device)].T
    return P, cost, col_ind


def match_tensors_sinkhorn_cc(r=0.5, get_merge_value=False,
                              print_costs=False, no_absval=False,
                              correlation_matrix=None,
                              sinkhorn_iters=20, num_opt_steps=300,
                              lr=0.01, tau_max=1.0, tau_min=0.01,
                              seed=42, **kwargs):
    """
    G4.1 Sinkhorn-CC: Differentiable permutation alignment using Sinkhorn
    relaxation and cross-correlation objective.

    Drop-in replacement for match_tensors_permute(). Uses gradient-based
    optimization of soft permutation matrices instead of one-shot Hungarian.

    Returns: (merge.T, unmerge, None, cost) — same format as match_tensors_permute()
    """
    if correlation_matrix is None:
        raise ValueError("correlation_matrix is required for match_tensors_sinkhorn_cc")

    O = correlation_matrix.shape[0]
    N = int(1 / (1 - r) + 0.5)
    Om = O // N
    device = correlation_matrix.device

    print(f"  Sinkhorn-CC: D={Om}, iters={sinkhorn_iters}, steps={num_opt_steps}, "
          f"lr={lr}, tau={tau_max}→{tau_min}")

    cross_corr = correlation_matrix[:Om, Om:2*Om]

    P, cost, col_ind = _sinkhorn_optimize(
        cross_corr, Om, device, no_absval=no_absval,
        sinkhorn_iters=sinkhorn_iters, num_opt_steps=num_opt_steps,
        lr=lr, tau_max=tau_max, tau_min=tau_min, seed=seed,
        print_costs=print_costs
    )

    mats = [torch.eye(Om, device=device), P]
    unmerge = torch.cat(mats, dim=0)
    merge = torch.cat(mats, dim=0)

    if print_costs:
        print(f"  Sinkhorn-CC final cost: {cost / (Om * 2):.6f}")

    if get_merge_value:
        merge_value = cost / (Om * 2)
        return merge.T, unmerge, merge_value

    return merge.T, unmerge, None, cost / (Om * 2)


def match_tensors_sinkhorn_cc_MHA(n_heads, permute_heads=False,
                                   head_assignments=[], r=0.5,
                                   get_merge_value=False, print_costs=True,
                                   no_absval=False, correlation_matrix=None,
                                   sinkhorn_iters=20, num_opt_steps=300,
                                   lr=0.01, tau_max=1.0, tau_min=0.01,
                                   seed=42, **kwargs):
    """
    G4.1 Sinkhorn-CC for multi-head attention nodes.
    Block-diagonal Sinkhorn per attention head.

    Returns: (merge.T, unmerge, head_assignments_or_None, cost)
    """
    if correlation_matrix is None:
        raise ValueError("correlation_matrix is required")

    O = correlation_matrix.shape[0]
    N = int(1 / (1 - r) + 0.5)
    Om = O // N
    device = correlation_matrix.device
    head_dim = Om // n_heads

    print(f"  Sinkhorn-CC MHA: D={Om}, heads={n_heads}, head_dim={head_dim}")

    # Step 1: Head-level assignment (Hungarian, same as G1)
    if permute_heads and not head_assignments:
        head_costs = np.zeros((n_heads, n_heads))
        for j in range(n_heads):
            for k in range(n_heads):
                a_slice = slice(head_dim * j, head_dim * (j + 1))
                b_slice = slice(Om + head_dim * k, Om + head_dim * (k + 1))
                sub_corr = correlation_matrix[a_slice, b_slice].cpu().numpy()
                if not no_absval:
                    sub_corr = np.absolute(sub_corr)
                head_costs[j, k] = sub_corr.sum()

        _, outer_col_ind = linear_sum_assignment(head_costs, maximize=True)
        head_assignments = outer_col_ind.tolist()
        print(f"    Head assignment: {head_assignments}")
    elif not head_assignments:
        head_assignments = list(range(n_heads))

    # Step 2: Per-head Sinkhorn optimization
    full_perm = torch.zeros(Om, Om, device=device)
    total_cost = 0

    for h_idx in range(n_heads):
        h_a = h_idx
        h_b = head_assignments[h_idx]

        a_slice = slice(head_dim * h_a, head_dim * (h_a + 1))
        b_slice = slice(Om + head_dim * h_b, Om + head_dim * (h_b + 1))
        head_cross_corr = correlation_matrix[a_slice, b_slice]

        P_head, head_cost, _ = _sinkhorn_optimize(
            head_cross_corr, head_dim, device, no_absval=no_absval,
            sinkhorn_iters=sinkhorn_iters, num_opt_steps=num_opt_steps,
            lr=lr, tau_max=tau_max, tau_min=tau_min, seed=seed + h_idx,
            print_costs=False
        )

        # Place in the full block-diagonal permutation
        row_start = head_dim * h_a
        col_start = head_dim * h_b
        full_perm[row_start:row_start + head_dim, col_start:col_start + head_dim] = P_head
        total_cost += head_cost

    avg_cost = total_cost / n_heads
    if print_costs:
        print(f"    Sinkhorn-CC MHA avg cost: {avg_cost:.4f}")

    mats = [torch.eye(Om, device=device), full_perm.T]
    unmerge = torch.cat(mats, dim=0)
    merge = torch.cat(mats, dim=0)

    if get_merge_value:
        return merge.T, unmerge, avg_cost
    return merge.T, unmerge, head_assignments, avg_cost