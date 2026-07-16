# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from itertools import chain

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.functional import interpolate
from typing import Tuple, Union


class Interpolation(nn.Module):
    """Interpolation nn.Module wrap for nn.functional.interpolate.

    Attributes:
        target_size (Tuple[int, int] | torch.Size): target spatial size of this interpolation.
    """

    def __init__(self, target_size: Union[Tuple[int, int], torch.Size]) -> None:
        super().__init__()
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Very simple forward pass to call interpolate()."""
        return interpolate(x, self.target_size)


class LightConvAdapterHead(nn.Module):
    """Light Convolutional Adapter module.

    Transforms features from source size [B, T, H, C] to target size [B, T, H, C].
    Uses CNN to map channel and spatial sizes jointly, while preserving the overall shape.
    """

    def __init__(
        self,
        source_size: Union[Tuple[int, ...], torch.Size],
        target_model: str,
        hidden_size_factor: Union[int, float] = 1.0,
    ):
        """Initialization function for LightConvAdapterHead.

        Args:
            source_size (Union[Tuple[int, ...], torch.Size]): the size of the source feature.
            hidden_size_factor (Union[int, float]): the size of hidden dim of feature translator
                as a factor of input feature hidden dim.
        """
        super().__init__()

        self.hidden_size_factor = hidden_size_factor
        source_channel_size = source_size[-1]  # Feature dimension (C)
        hidden_dim = int(source_channel_size * hidden_size_factor)  # Adjusting based on the channel dimension
        self.source_size = source_size
        self.target_length = 1214
        self.target_model = target_model
        # Define the convolutional adapter
        self.adapter = nn.Sequential(
            nn.LayerNorm(source_channel_size),  # Apply LayerNorm over the channel dimension
            Rearrange('b t h c -> b c h t'),  # Permute to [B, C, H, T]
            nn.Conv2d(source_channel_size, hidden_dim, kernel_size=(1, 3), padding=(0, 1)),  # Conv maintains shape
            nn.ReLU(),
            Rearrange('b c h t -> b t h c'),
            nn.LayerNorm(hidden_dim),  # Apply LayerNorm on the hidden dimension
            Rearrange('b c h t -> b t h c'),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1)),  # Maintains shape
            nn.ReLU(),
            Rearrange('b c h t -> b t h c'),
            nn.LayerNorm(hidden_dim),  # Apply LayerNorm on the hidden dimension
            Rearrange('b c h t -> b t h c'),
            nn.Conv2d(hidden_dim, source_channel_size, kernel_size=(1, 3), padding=(0, 1)),  # Return to original size
            Rearrange('b c h t -> b t h c'),
        )
        
        self.length_adjuster = nn.Sequential(
            nn.LayerNorm(source_channel_size),  # Apply LayerNorm over the last dimension
            
            Rearrange('b c h t -> b t h c'),
            nn.Conv2d(source_channel_size, source_channel_size, kernel_size=(1, 3), padding=(0, 1)),  # Conv along the T dimension
            Rearrange('b c h t -> b t h c'),

            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.target_length, source_channel_size)),  # Resize to target length
        )
        
        self.adapter.register_full_backward_hook(self.grad_divide_hook)
        self.length_adjuster.register_full_backward_hook(self.grad_divide_hook)
        
    def grad_divide_hook(self, module, grad_input, grad_output):
        """Hook function to divide the gradient by a factor (e.g., 5)."""
        factor = 5  # The factor by which to divide the gradients
        # Make sure the number of grad_input elements matches exactly what is expected
        if len(grad_input) == 1:  # Typically for nn.Linear
            return (grad_input[0] / factor,)  # Return as a single-element tuple
        else:
            return tuple(grad / factor if grad is not None else None for grad in grad_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for LightConvAdapterHead.

        Args:
            x (torch.Tensor): input tensor with shape [B, T, H, C].

        Returns:
            torch.Tensor: transformed tensor matching the input shape.
        """
        # Permute to [B, C, H, T] to apply Conv2d correctly
        # x = x.permute(0, 3, 2, 1)  # Permute to [B, C, H, T]
        # print("-----------------------------------------------",x.shape, self.source_size)
        # Pass through the adapter which preserves the shape
        x = self.adapter(x)

        if self.target_model == 'ast':
            x = self.length_adjuster(x)
        # Permute back to original shape [B, T, H, C]
        # x = x.permute(0, 3, 2, 1)  # Back to [B, T, H, C]

        return x

