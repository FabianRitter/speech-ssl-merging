import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
from .dictionary import Dictionary
import logging
import os
import random
#-------------#
import pandas as pd
from tqdm import tqdm
from pathlib import Path
#-------------#
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
#-------------#
import torchaudio
#-------------#
from .dictionary import Dictionary

SAMPLE_RATE = 16000
HALF_BATCHSIZE_TIME = 2000

    
# Interface
def get_dataloader(config, split="train"):
    """
    Args:
        split: string
            The name of the dataloader, can be train/dev/test-clean/test-other for asr

    Return:
        a torch.utils.data.DataLoader returning each batch in the format of:

        [wav1, wav2, ...], your_other_contents1, your_other_contents2, ...

        where wav1, wav2 ... are in variable length
        each wav is torch.FloatTensor in cpu with:
            1. dim() == 1
            2. sample_rate == 16000
            3. directly loaded by torchaudio
    """
    dictionary_path = config["datarc"]["dict_path"]
    dictionary = Dictionary.load(dictionary_path)
    train_dataset = SequenceDataset(
                split, 
                config["datarc"]["train_batch_size"], 
                dictionary, 
                upstream="hubert_base",
                features_path=None, 
                **config["datarc"]
            )

    if split == "train":
        return _get_train_dataloader(train_dataset, config)


def _get_train_dataloader(dataset, config):
    sampler = DistributedSampler(dataset) if is_initialized() else None
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config["datarc"].get("num_workers", 1),
        collate_fn=dataset.collate_fn,
    )

class SequenceDataset(Dataset):
    
    def __init__(self, split, bucket_size, dictionary, libri_root, bucket_file, **kwargs):
        super(SequenceDataset, self).__init__()
        
        self.dictionary = dictionary
        self.libri_root = libri_root
        self.sample_rate = SAMPLE_RATE
        self.split_sets = kwargs[split]
        self.upstream_name = kwargs['upstream']
        self.features_path = None

        # Read table for bucketing
        assert os.path.isdir(bucket_file), 'Please first run `python3 preprocess/generate_len_for_bucket.py -h` to get bucket file.'

        # Wavs
        table_list = []
        for item in self.split_sets:
            file_path = os.path.join(bucket_file, item + ".csv")
            if os.path.exists(file_path):
                table_list.append(
                    pd.read_csv(file_path)
                )
            else:
                logging.warning(f'{item} is not found in bucket_file: {bucket_file}, skipping it.')

        table_list = pd.concat(table_list)
        table_list = table_list.sort_values(by=['length'], ascending=False)

        X = table_list['file_path'].tolist()
        X_lens = table_list['length'].tolist()

        assert len(X) != 0, f"0 data found for {split}"

        # Transcripts

        x_names = set([self._parse_x_name(x) for x in X])

        self.Y = {}  # We'll use __getitem__ to compute labels on the fly.


        # Use bucketing to allow different batch sizes at run time
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in tqdm(zip(X, X_lens), total=len(X), desc=f'ASR dataset {split}', dynamic_ncols=True):
            batch_x.append(x)
            batch_len.append(x_len)
            
            # Fill in batch_x until batch is full
            if len(batch_x) == bucket_size:
                # Half the batch size if seq too long
                if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME):
                    self.X.append(batch_x[:bucket_size//2])
                    self.X.append(batch_x[bucket_size//2:])
                else:
                    self.X.append(batch_x)
                batch_x, batch_len = [], []
        
        # Gather the last batch
        if len(batch_x) > 1:
            self.X.append(batch_x)

    def _parse_x_name(self, x):
        return x.split('/')[-1].split('.')[0]

    def _load_wav(self, wav_path):
        wav, sr = torchaudio.load(os.path.join(self.libri_root, wav_path))
        assert sr == self.sample_rate, f'Sample rate mismatch: real {sr}, config {self.sample_rate}'
        return wav.reshape(-1)


    def _build_dictionary(self, transcripts, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        d = Dictionary()
        transcript_list = list(transcripts.values())
        Dictionary.add_transcripts_to_dictionary(
            transcript_list, d, workers
        )
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d


    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        wav_batch = [self._load_wav(x_file).numpy() for x_file in self.X[index]]
        label_batch = [len(wav) for wav in wav_batch]
        filename_batch = [Path(x_file).stem for x_file in self.X[index]]
        if self.features_path:
            feature = []
            fname_or_true = []
            for idx, fname in enumerate(filename_batch):
                feature_path = os.path.join(self.features_path, self.upstream_name, f"{fname}.pt")
                if os.path.exists(feature_path):
                    feature.append(torch.load(feature_path))
                    fname_or_true.append(True)
                else:
                    feature.append(wav_batch[idx])
                    fname_or_true.append(filename_batch[idx])
                return feature, label_batch, fname_or_true
        return wav_batch, label_batch, filename_batch # bucketing, return ((wavs, labels))

    def collate_fn(self, items):
        assert len(items) == 1
        return items[0][0], items[0][1], items[0][2] # hack bucketing, return (wavs, labels, filenames)
