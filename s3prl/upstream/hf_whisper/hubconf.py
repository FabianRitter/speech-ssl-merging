from .expert import UpstreamExpert as _UpstreamExpert


def hf_whisper_custom(ckpt, *args, **kwargs):
    return _UpstreamExpert(ckpt, *args, **kwargs)

def whisper_large_v3(refresh=False, *args, **kwargs):
    kwargs["ckpt"] = "openai/whisper-large-v3"
    return hf_whisper_custom(refresh=refresh, *args, **kwargs)

def whisper_large_v2(refresh=False, *args, **kwargs):
    kwargs["ckpt"] = "openai/whisper-large-v2"
    return hf_whisper_custom(refresh=refresh, *args, **kwargs)

def whisper_large(refresh=False, *args, **kwargs):
    kwargs["ckpt"] = "openai/whisper-large"
    return hf_whisper_custom(refresh=refresh, *args, **kwargs)

def whisper_medium(refresh=False, *args, **kwargs):
    kwargs["ckpt"] = "openai/whisper-medium"
    return hf_whisper_custom(refresh=refresh, *args, **kwargs)

def whisper_small(refresh=False, *args, **kwargs):
    kwargs["ckpt"] = "openai/whisper-small"
    return hf_whisper_custom(refresh=refresh, *args, **kwargs)

def whisper_base(refresh=False, *args, **kwargs):
    kwargs["ckpt"] = "openai/whisper-base"
    return hf_whisper_custom(refresh=refresh, *args, **kwargs)

def whisper_tiny(refresh=False, *args, **kwargs):
    kwargs["ckpt"] = "openai/whisper-tiny"
    return hf_whisper_custom(refresh=refresh, *args, **kwargs)
