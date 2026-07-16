MODEL_FEATURE_SIZES = {
    "hubert_base": (12, 711, 768),
    "mert_v0_public": (12, 711, 768),
    "ast": (12, 1214, 768),
    "atst_base": (12, 1214, 768),
    "whisper_large": (1024, 16, 16),
}


def get_model_feature_size(
    model_name: str, return_torch_size: bool = False
) -> tuple[int, ...] | torch.Size:
    """
    Get the size of queried model feature.

    Args:
        model_name (str): name of the model.
        return_torch_size (bool): return torch.Size instead of python tuple. Defaults to False.

    Returns:
        tuple[int, ...] | torch.Size: the size of the feature.
    """
    size: tuple[int, ...] = MODEL_FEATURE_SIZES[model_name]

    if return_torch_size:
        size = torch.Size(size)

    return size