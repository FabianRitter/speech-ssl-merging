import torch
from abc import ABC, abstractmethod
import pdb

class MetricCalculator(ABC):
    
    @abstractmethod
    def update(self, batch_size, dx, *feats, **aux_params): return NotImplemented
    
    @abstractmethod
    def finalize(self): return NotImplemented


def compute_correlation(covariance, eps=1e-7):
    # Ensure covariance diagonal is non-negative before computing std dev
    variances = torch.diagonal(covariance)
    # Clamp small negatives to zero OR add small epsilon to ensure positivity
    # Let's add epsilon as it also helps stabilize the division
    stabilized_variances = variances + eps # Add epsilon directly    
    std = stabilized_variances.sqrt()
    # Prevent division by zero if std dev is extremely small
    std_clamped = torch.clamp(std, min=eps)
    # Use outer product of clamped std dev for normalization
    correlation = covariance / torch.outer(std_clamped, std_clamped)
    # Clamp final correlation values to [-1, 1] just in case of numerical overshoot
    correlation = torch.clamp(correlation, min=-1.0, max=1.0)
    return correlation

class CovarianceMetric(MetricCalculator):
    name = 'covariance'

    def __init__(self):
        self.mean = None
        self.outer = None
        # Removed self.sos as it's implicitly in self.outer's diagonal
        self.total_feature_vectors = 0 # Keep track of the exact number of samples
        self.feature_dim = None # Store feature dimension
        # Removed self.bsz and self.num_updates as they aren't used in finalize

    def update(self, batch_size, *feats, **aux_params):
        # Concatenate features from all models for this batch
        # feats dimensions: model1[dim, n_vecs_1], model2[dim, n_vecs_2], ...
        # After cat(dim=0): [num_models * dim, n_vecs_batch] (assuming same n_vecs per model in batch)
        # Let's assume feats come in as a list: [model1_feats, model2_feats]
        # model1_feats shape: [feature_dim, num_vectors_in_batch_model1]
        # model2_feats shape: [feature_dim, num_vectors_in_batch_model2]
        # We need to concatenate along feature dimension for covariance across models
        # Final expected shape for calculation: [TotalFeatures, NumVectors]
        # TotalFeatures = num_models * feature_dim
        
        if not feats: return # Skip if no features provided

        # Check shapes and concatenate along feature dimension (dim=0)
        # Assuming all feature tensors in feats tuple have the same number of columns (vectors)
        # and the same number of rows per model (feature_dim)
        num_vectors_batch = feats[0].shape[1]
        if self.feature_dim is None:
            self.feature_dim = feats[0].shape[0] # feature dim per model
        
        # Verify shapes and concatenate
        processed_feats = []
        for f in feats:
            if f.shape[0] != self.feature_dim:
                 print(f"Warning: Feature dimension mismatch in CovarianceMetric update. Expected {self.feature_dim}, got {f.shape[0]}. Skipping batch.")
                 # pdb.set_trace() # Debug here if needed
                 return
            if f.shape[1] != num_vectors_batch:
                 print(f"Warning: Number of vectors mismatch in CovarianceMetric update. Expected {num_vectors_batch}, got {f.shape[1]}. Skipping batch.")
                 # pdb.set_trace() # Debug here if needed
                 return
            processed_feats.append(f)

        # Concatenate features along the feature dimension (rows)
        combined_feats = torch.cat(processed_feats, dim=0) # Shape: [TotalFeatures, NumVectors]
        combined_feats = torch.nan_to_num(combined_feats, 0., 0., 0.) # Handle potential NaNs

        # Accumulate statistics using float64 for precision
        current_mean = combined_feats.sum(dim=1) # Sum across vectors for each feature
        current_outer = combined_feats @ combined_feats.T # Outer product sum

        if self.mean is None:
            # Initialize accumulators on first update
            total_feature_dim = combined_feats.shape[0]
            self.mean = torch.zeros(total_feature_dim, dtype=torch.float64, device=combined_feats.device)
            self.outer = torch.zeros((total_feature_dim, total_feature_dim), dtype=torch.float64, device=combined_feats.device)

        self.mean += current_mean.to(torch.float64)
        self.outer += current_outer.to(torch.float64)
        self.total_feature_vectors += num_vectors_batch # Accumulate exact number of vectors

    def finalize(self, eps=1e-9, print_featnorms=False, **kwargs): # Removed numel, dot_prod, pca, scale_cov, normalize args for clarity
        if self.total_feature_vectors == 0:
            print("Warning: CovarianceMetric finalize called with zero samples.")
            # Return identity or zero matrix? Let's return None to indicate failure.
            # dim = self.mean.shape[0] if self.mean is not None else 0
            # return torch.zeros((dim, dim), device=self.mean.device) if dim > 0 else None
            return None # Indicate no data

        N = self.total_feature_vectors

        # Calculate E[X] and E[X X^T]
        E_X = self.mean / N
        E_XXT = self.outer / N

        # Calculate Covariance: E[X X^T] - E[X] E[X]^T
        cov = E_XXT - torch.outer(E_X, E_X)

        # --- Numerical Stability ---
        # Add a small epsilon to the diagonal to ensure positive definiteness
        # and stability for correlation calculation.
        cov.diagonal().add_(eps)

        # --- Sanity Check ---
        # Check for negative variances *after* adding epsilon (shouldn't happen now)
        variances = torch.diagonal(cov)
        if (variances < 0).any():
             num_negative = (variances < 0).sum()
             print(f"WARNING: Found {num_negative} negative variances on diagonal AFTER adding epsilon! Clamping them to {eps}.")
             print(f"Min variance found: {variances.min()}")
             # Force clamp if the epsilon addition wasn't enough (shouldn't be needed often)
             cov.diagonal().copy_(torch.clamp(variances, min=eps))
             # pdb.set_trace() # Investigate if this happens frequently

        # --- Optional: Print Feature Norms (Needs self.outer diagonal) ---
        if print_featnorms:
             # E[X^2] is on the diagonal of E_XXT
             sum_of_squares_per_feature = torch.diagonal(self.outer)
             rms_per_feature = (sum_of_squares_per_feature / N).sqrt() # RMS value
             
             num_models = len(self.mean) // self.feature_dim
             print("Feature RMS norms per model:")
             for i in range(num_models):
                  start_idx = i * self.feature_dim
                  end_idx = (i + 1) * self.feature_dim
                  model_rms = rms_per_feature[start_idx:end_idx]
                  print(f"  Model {i+1}: Mean RMS={torch.mean(model_rms).item():.4f}, Std RMS={torch.std(model_rms).item():.4f}")


        # Return the stabilized covariance matrix (float32 is usually sufficient)
        return cov.float()

class MeanMetric(MetricCalculator):
    name = 'mean'
    
    def __init__(self):
        self.mean = None
    
    def update(self, batch_size, *feats, **aux_params):
        feats = torch.cat(feats, dim=0)
        mean = feats.abs().mean(dim=1)
        if self.mean is None: 
            self.mean = torch.zeros_like(mean)
        self.mean  += mean  * batch_size
    
    def finalize(self, numel, eps=1e-4, print_featnorms=False):
        return self.mean / numel
        

def get_metric_fns(names):
    metrics = {}
    for name in names:
        if name == 'mean':
            metrics[name] = MeanMetric
        elif name == 'covariance':
            metrics[name] = CovarianceMetric
        else:
            raise NotImplementedError(name)
    return metrics