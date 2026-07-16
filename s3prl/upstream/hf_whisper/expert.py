import logging

import torch
from transformers import AutoFeatureExtractor, WhisperModel, WhisperFeatureExtractor
from ..interfaces import UpstreamBase
import time

SAMPLE_RATE = 16000

logger = logging.getLogger(__name__)


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwds):
        super().__init__()
        # self.extracter = AutoFeatureExtractor.from_pretrained(ckpt)
        self.extracter = WhisperFeatureExtractor.from_pretrained(ckpt)
        self.model = WhisperModel.from_pretrained(ckpt).get_encoder()
        self.model._freeze_parameters()

    def get_downsample_rates(self, key: str = None) -> int:
        return 160

    def forward(self, wavs):
        device = wavs[0].device
        wavs = [wav.detach().cpu().numpy() for wav in wavs]
        input = self.extracter(
            wavs,
            return_tensors = "pt",
            return_attention_mask = False,
            sampling_rate = SAMPLE_RATE,
            device = device
        ).to(device)
        outputs = self.model(input.input_features, output_hidden_states=True, return_dict=True)
        return outputs
