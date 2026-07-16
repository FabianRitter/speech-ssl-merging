import math
from typing import Any, Optional, Dict, Tuple, List, Union

import torch
import torch.nn as nn

from .adapter_heads import LightConvAdapterHead


class FeatureTranslator(nn.Module):
    """Base class for the feature translator.

    The flow is backbone_adapter -> translator_stem -> translator_heads.

    Attributes:
        backbone_feature_size (torch.Size): the size of features of the backbone.
        target_feature_sizes (Dict[str, Union[torch.Size, Tuple[int, ...]]]): the sizes of features of target models.
        translator_hidden_size (int): the hidden dim of the translator. Defaults to 2048.
        target_model_names (List[str]): convenient attribute to hold all the names of the target models.

        backbone_adapter (nn.Module): the adapter to map channel dim of backbone to the translator hidden dim.
        translator_stem (nn.Module):  the shared stem for all target models.
        translator_heads (nn.ModuleDict): specific heads for different target models.
    """

    def __init__(
        self,
        # backbone_feature_size: torch.Size,
        # target_feature_sizes: Dict[str, Union[torch.Size, Tuple[int, ...]]],
        target_model_names: list,
        input_size: Tuple[int, int, int, int],
        translator_hidden_size: int = 1024,
    ) -> None:
        """Initialization function for FeatureTranslator.

        Args:
            backbone_feature_size (torch.Size): the size of features of the backbone.
            target_feature_sizes (Dict[str, Union[torch.Size, Tuple[int, ...]]]): the sizes of features of target models.
            translator_hidden_size (int): the hidden dim of the translator. Defaults to 2048.
        """
        super().__init__()
        # self.backbone_feature_size = backbone_feature_size  # (C, H, W)
        # self.target_feature_sizes = target_feature_sizes  # [(C, H, W)]
        self.input_channel = input_size[-1]
        self.translator_hidden_size = translator_hidden_size  # C
        # self.legit_target_model_name_map: Dict[str, str] = {t: t.replace(".", "_") for t in self.target_model_names}
        self.translator_heads: nn.ModuleDict = None
        self.target_model_names = target_model_names
        self.input_size = input_size
        self.backbone_adapter = nn.Sequential(
            nn.LayerNorm(self.input_channel),  # do a pre-norm
            nn.Linear(
                self.input_channel,  # C in [C,H,W]
                self.translator_hidden_size,
            ),
        )
        self.translator_stem: nn.Module = nn.Identity()
        self.build_translator_heads()

    def build_translator_heads(self) -> None:
        """Build translator heads to match the dimension of each target feature set.

        Example:
            translator_heads: Dict[str, nn.Module] = ...
            self.translator_heads = nn.ModuleDict(translator_heads)
        """
        raise NotImplementedError("build_translator_heads() should be overridden")

    def forward(
        self, x: torch.Tensor, target_model_names: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for a base feature translator.

        Args:
            x (torch.Tensor): input features from the backbone. [B, (1)+H*W, C].
                (1) means optional CLS token. If `backbone_no_cls==True`, then [B, H*W, C].
            target_model_names (Optional[List[str]]): names of the target models.

        Returns:
            Dict[str, torch.Tensor]: predicted features for target models.
        """
        # x: [B, (1)+H*W, C]
        x = self.backbone_adapter(x)  
        x = self.translator_stem(x) 
        target_model_names = target_model_names if target_model_names is not None else self.target_model_names
        features = {t: self.translator_heads[t](x) for t in target_model_names}
        return features


class LightConvFeatureTranslator(FeatureTranslator):
    def __init__(
        self,
        target_model_names: list,
        input_size: Tuple[int, int, int, int],
        translator_hidden_size: int = 1024,
        hidden_size_factor: Union[int, float] = 1.0,
    ) -> None:
        """Initialization function for LightConvFeatureTranslator.
            It's for a smaller translator compared to ConvFeatureTranslator.

        Args:
            backbone_feature_size (torch.Size): the size of features of the backbone.
            target_feature_sizes (Dict[str, Union[torch.Size, Tuple[int, ...]]]): the sizes of features of target models.
            translator_hidden_size (Optional[int]): the hidden dim of the translator. Defaults to 1024.
            hidden_size_factor: the size of hidden dim of feature translator
                as a factor of input feature hidden dim. Defaults to 1.0
        """
        self.hidden_size_factor = hidden_size_factor
        super().__init__(
            target_model_names = target_model_names,
            input_size=input_size,
            translator_hidden_size=translator_hidden_size,
        )
        self.backbone_adapter = nn.Identity()

    def build_translator_heads(self) -> None:
        """Build translator heads to match the dimension of each target feature set."""
        translator_heads = {}
        for target_model in self.target_model_names:
            # if target_model == "ast":
            #     target_size = [input_size[0], input_size[1], 1214, input_size[3]]  # [4, 2, 1214, 768]
            # else:
            #     target_size = self.input_size

            head = LightConvAdapterHead(
                source_size=self.input_size, 
                target_model=target_model,
                hidden_size_factor=self.hidden_size_factor
            )

            translator_heads[target_model] = head
        self.translator_heads = nn.ModuleDict(translator_heads)


def build_feature_translator(translator_type: str, **kwargs: Any) -> FeatureTranslator:
    """Handy function to build feature translators given the type

    Args:
        translator_type (str): the type of the translator,
            one in `"mlp"`, `"conv"`, `"lconv"`, `"transformer"` (or `"trans"`).
            At the moment we are actively using `"lconv"`.

    Returns:
        FeatureTranslator: the corresponding FeatureTranslator
    """
    if translator_type == "mlp":
        return MLPFeatureTranslator(**kwargs)
    elif translator_type == "conv":
        return ConvFeatureTranslator(**kwargs)
    elif translator_type == "lconv":
        return LightConvFeatureTranslator(**kwargs)
    elif translator_type == "transformer" or translator_type == "trans":
        return TransformerFeatureTranslator(**kwargs)
    else:
        raise NotImplementedError(f"Requested {translator_type} is not implemented yet.")
