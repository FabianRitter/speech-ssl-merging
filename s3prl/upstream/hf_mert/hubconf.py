from .expert import UpstreamExpert as _UpstreamExpert


def hf_mert_custom(ckpt, *args, **kwargs):
    return _UpstreamExpert(ckpt, *args, **kwargs)

def mert_v1_330m(refresh=False, *args, **kwargs):
    kwargs["ckpt"] = "m-a-p/MERT-v1-330M"
    return hf_mert_custom(refresh=refresh, *args, **kwargs)

def mert_v1_95m(refresh=False, *args, **kwargs):
    kwargs["ckpt"] = "m-a-p/MERT-v1-95M"
    return hf_mert_custom(refresh=refresh, *args, **kwargs)

def mert_v0_public(refresh=False, *args, **kwargs):
    kwargs["ckpt"] = "m-a-p/MERT-v0-public"
    return hf_mert_custom(refresh=refresh, *args, **kwargs)

def mert_v0(refresh=False, *args, **kwargs):
    kwargs["ckpt"] = "m-a-p/MERT-v0"
    return hf_mert_custom(refresh=refresh, *args, **kwargs)

def music2vec_v1(refresh=False, *args, **kwargs):
    kwargs["ckpt"] = "m-a-p/music2vec-v1"
    return hf_mert_custom(refresh=refresh, *args, **kwargs)
