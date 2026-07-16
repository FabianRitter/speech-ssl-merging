import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized

from argparse import Namespace
from pathlib import Path

from ..model import *
from .dataset import GtzanDataset

class DownstreamExpert(nn.Module):
    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        """
        Initialize the downstream expert for GTZAN genre classification.
        
        Args:
            upstream_dim (int): Dimension of upstream model features.
            downstream_expert (dict): Config for downstream expert (datarc, modelrc).
            expdir (str): Experiment directory.
            **kwargs: Additional arguments.
        """
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.expdir = expdir
        self.pre_extract_dir = kwargs.get("pre_extract_dir", None)
        self.best_score = 0

        # Initialize datasets and dataloaders
        root_dir = Path(self.datarc['file_path'])
        self.train_dataset = GtzanDataset(split='train', root_dir=str(root_dir), target_sr=kwargs.get("sample_rate", 24000),
            distortion_mode=self.datarc.get('distortion_mode'),
            distortion_types=self.datarc.get('distortion_types'),
            distortion_config=self.datarc.get('distortion_config'),
        )
        self.dev_dataset = GtzanDataset(split='valid', root_dir=str(root_dir), target_sr=kwargs.get("sample_rate", 24000),
            distortion_mode=self.datarc.get('distortion_mode'),
            distortion_types=self.datarc.get('distortion_types'),
            distortion_config=self.datarc.get('distortion_config'),
        )
        self.test_dataset = GtzanDataset(split='test', root_dir=str(root_dir), target_sr=kwargs.get("sample_rate", 24000),
            distortion_mode=self.datarc.get('distortion_mode'),
            distortion_types=self.datarc.get('distortion_types'),
            distortion_config=self.datarc.get('distortion_config'),
        )



        # Initialize downstream model (10 genres)
        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.dropout = nn.Dropout(p=0.3)
        self.model = model_cls(
            input_dim = self.modelrc['projector_dim'],
            output_dim = len(self.train_dataset.label_to_idx.keys()),
            **model_conf,
        )

        self.objective = nn.CrossEntropyLoss()
        self.register_buffer('best_acc', torch.zeros(1))

    
    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'], 
            shuffle=(sampler is None), sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_valid_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)
    
    def get_dataloader(self, mode):
        return eval(f'self.get_{mode}_dataloader')()


    def forward(self, mode, features, labels, filenames, records, **kwargs):
        """
        Process a batch through upstream and downstream models.
        
        Args:
            mode (str): 'train', 'dev', or 'test'.
            batch (dict): Batched data from dataloader.
            records (dict): For logging metrics.
        
        Returns:
            torch.Tensor: Loss value.
        """
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)
        features = pad_sequence(features, batch_first=True)
        features = self.projector(features)
        features = self.dropout(features)
        predicted, _ = self.model(features, features_len)
        labels = torch.LongTensor(labels).to(features.device)
        loss = self.objective(predicted, labels)

        predicted_classid = predicted.max(dim=-1).indices
        records['acc'] += (predicted_classid == labels).view(-1).cpu().float().tolist()
        records['loss'].append(loss.item())
        idx_to_genre = {idx: genre for genre, idx in self.train_dataset.label_to_idx.items()}

        records['predicted_genre'] += [idx_to_genre[i] for i in predicted_classid.cpu().tolist()]
        records['truth_genre'] += [idx_to_genre[i] for i in labels.cpu().tolist()]
        records['filename'] += filenames

        return loss

     # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        for key in ["acc", "loss"]:
            average = torch.FloatTensor(records[key]).mean().item()
            logger.add_scalar(
                f'gtzan_genre/{mode}-{key}',
                average,
                global_step=global_step
            )
            with open(Path(self.expdir) / "log.log", 'a') as f:
                if key == 'acc':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} at step {global_step}: {average}\n')
                    if mode == 'valid' and average > self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(f'New best on {mode} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-best.ckpt')

        if mode in ["valid", "test"]:
            with open(Path(self.expdir) / f"{mode}_predict.txt", "w") as file:
                lines = [f"{f} {p}\n" for f, p in zip(records["filename"], records["predicted_genre"])]
                file.writelines(lines)

            with open(Path(self.expdir) / f"{mode}_truth.txt", "w") as file:
                lines = [f"{f} {l}\n" for f, l in zip(records["filename"], records["truth_genre"])]
                file.writelines(lines)


        return save_names