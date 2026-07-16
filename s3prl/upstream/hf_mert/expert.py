import logging

import torch
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
from ..interfaces import UpstreamBase
import time

logger = logging.getLogger(__name__)


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwds):
        super().__init__()
        # self.extracter = AutoFeatureExtractor.from_pretrained(ckpt)
        self.extracter = Wav2Vec2FeatureExtractor.from_pretrained(ckpt, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(ckpt, trust_remote_code=True)
        # self.model._freeze_parameters()
        self.sample_rate = 16000 if "MERT-v0" in ckpt or "music2vec" in ckpt else 24000

    def get_downsample_rates(self, key: str = None) -> int:
        return 320

    def forward(self, wavs):
        device = wavs[0].device
        wavs = [wav.detach().cpu().numpy() for wav in wavs]
        processed_input = self.extracter(
            wavs,
            return_tensors = "pt",
            sampling_rate = self.sample_rate,
            padding = True,
            device = device
        ).to(device)
        outputs = self.model(**processed_input, output_hidden_states=True, return_dict=True)
        return outputs
