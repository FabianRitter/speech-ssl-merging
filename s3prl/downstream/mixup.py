import numpy as np
import torch
import torch.nn.functional as F

def mixup_target(target, lam=1., smoothing=0.0, num_classes=None):
    """
    Transform target to one-hot and apply mixup.
    
    Args:
        target (Tensor or Tuple): Class indices of shape [batch_size].
        lam (float or Tensor): Mixing factor.
        smoothing (float): Label smoothing value.
        num_classes (int): Number of classes (required for one-hot encoding).
    
    Returns:
        Tensor: Mixed one-hot targets.
    """
    if isinstance(target, tuple):
        target = torch.tensor(target, dtype=torch.long)
    
    if num_classes is None:
        raise RuntimeError("Number of classes must be provided")  # Infer number of classes if not provided
    
    # Convert target to one-hot encoding
    target_one_hot = F.one_hot(target, num_classes=num_classes).float()

    # Apply label smoothing
    if smoothing > 0.0:
        target_one_hot = target_one_hot * (1.0 - smoothing) + (smoothing / num_classes)

    # Reverse the batch for mixing
    target_flipped = target_one_hot.flip(0)

    # Mix targets
    if isinstance(lam, torch.Tensor):
        lam = lam.view(-1, 1)  # Ensure compatibility with one-hot dimensions
    mixed_target = target_one_hot * lam + target_flipped * (1.0 - lam)

    return mixed_target


class Mixup:
    """ Mixup for a list of tensors with shape [batch, time, hidden]

    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        prob (float): probability of applying mixup per batch or element
        mode (str): how to apply mixup params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        label_smoothing (float): apply label smoothing to the mixed target tensor
    """
    def __init__(self, mixup_alpha=1., prob=1.0, mode='batch', label_smoothing=0.1):
        self.mixup_alpha = mixup_alpha
        self.mix_prob = prob
        self.label_smoothing = label_smoothing
        self.mode = mode
        self.mixup_enabled = True  # set to false to disable mixing (intended to be set by train loop)

    def _params_per_elem(self, batch_size):
        lam = np.ones(batch_size, dtype=np.float32)
        if self.mixup_enabled and self.mixup_alpha > 0.:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
            lam = np.where(np.random.rand(batch_size) < self.mix_prob, lam.astype(np.float32), 1.0)
        return lam

    def _mix_elem(self, x):
        batch_size = len(x)
        lam_batch = self._params_per_elem(batch_size)
        x_orig = [t.clone() for t in x]  # Clone each tensor in the list to keep the original
        for i in range(batch_size):
            j = batch_size - i - 1  # flip index for mixup
            lam = lam_batch[i]
            if lam != 1.:
                x[i] = x[i] * lam + x_orig[j] * (1 - lam)
        return torch.tensor(lam_batch, dtype=torch.float32).unsqueeze(1)

    def _mix_pair(self, x):
        batch_size = len(x)
        lam_batch = self._params_per_elem(batch_size // 2)
        x_orig = [t.clone() for t in x]  # Clone each tensor in the list to keep the original
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                x[i] = x[i] * lam + x_orig[j] * (1 - lam)
                x[j] = x[j] * lam + x_orig[i] * (1 - lam)
        lam_batch = np.concatenate((lam_batch, lam_batch[::-1]))
        return torch.tensor(lam_batch, dtype=torch.float32).unsqueeze(1)

    def _mix_batch(self, x):
        lam = 1.0
        if self.mixup_enabled and np.random.rand() < self.mix_prob and self.mixup_alpha > 0.:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        if lam == 1.:
            return lam
        x_flipped = [t.clone() for t in reversed(x)]
        for i in range(len(x)):
            x[i] = x[i] * lam + x_flipped[i] * (1 - lam)
        return lam

    def __call__(self, x, target, num_classes=None):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        if self.mode == 'elem':
            lam = self._mix_elem(x)
        elif self.mode == 'pair':
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        target = mixup_target(target, lam, self.label_smoothing, num_classes)
        return x, target