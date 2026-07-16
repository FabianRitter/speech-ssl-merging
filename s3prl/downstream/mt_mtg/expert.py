import torch
#-------------#
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
#-------------#
from ..model import *
from .dataset import MTGTop50AudioDataset, MTGTop50FeatureDataset
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.downstream = downstream_expert
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.expdir = expdir

        root_dir = Path(self.datarc['file_path'])
        self.pre_extract_dir = kwargs["pre_extract_dir"]
        
        self.train_dataset = MTGTop50FeatureDataset(self.pre_extract_dir, self.datarc, 'train') if self.pre_extract_dir else MTGTop50AudioDataset(root_dir, self.datarc, 'train')
        self.dev_dataset = MTGTop50FeatureDataset(self.pre_extract_dir, self.datarc, 'dev') if self.pre_extract_dir else MTGTop50AudioDataset(root_dir, self.datarc, 'dev')
        self.test_dataset = MTGTop50FeatureDataset(self.pre_extract_dir, self.datarc, 'test') if self.pre_extract_dir else MTGTop50AudioDataset(root_dir, self.datarc, 'test')
        
        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim = self.modelrc['projector_dim'],
            output_dim = len(self.train_dataset.class2id.keys()),
            **model_conf,
        )
        self.objective = nn.CrossEntropyLoss()
        self.register_buffer('best_score', torch.zeros(1))
        

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

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    # Interface
    def get_dataloader(self, mode):
        return eval(f'self.get_{mode}_dataloader')()

    # Interface
    def forward(self, mode, features, labels, filenames, records, **kwargs):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)
        features = pad_sequence(features, batch_first=True)
        features = self.projector(features)
        predicted, _ = self.model(features, features_len)
        labels = torch.stack(labels).to(features.device)
        loss = self.objective(predicted, labels)
        # Deal with AUC-ROC and PR-ROC
        records['logit'] += predicted.cpu().float().tolist()
        records['labels'] += labels.cpu().float().tolist()
        records['loss'].append(loss.item())
        records['filename'] += filenames
        # records['predict_top50'] += self.train_dataset.label2class(predicted_classid.cpu().tolist())
        # records['truth_top50'] += self.train_dataset.label2class(labels.cpu().tolist())

        return loss
    
    def compute_macro_average_roc_auc(self, y_true, y_scores):
        roc_auc_scores = []
        for i in range(y_true.shape[1]):
            roc_auc = roc_auc_score(y_true[:, i], y_scores[:, i])
            roc_auc_scores.append(roc_auc)
        return np.mean(roc_auc_scores)

    def compute_macro_average_pr_auc(self, y_true, y_scores):
        pr_auc_scores = []
        for i in range(y_true.shape[1]):
            pr_auc = average_precision_score(y_true[:, i], y_scores[:, i])
            pr_auc_scores.append(pr_auc)
        return np.mean(pr_auc_scores)
    
    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        roc_auc = self.compute_macro_average_roc_auc(np.array(records['labels']), np.array(records['logit']))
        pr_auc = self.compute_macro_average_pr_auc(np.array(records['labels']), np.array(records['logit']))
        results = {'roc_auc': roc_auc, 'pr_auc': pr_auc}
        for key, value in results.items():
            logger.add_scalar(
                f'mtg_top50/{mode}-{key}',
                value,
                global_step=global_step
            )
            with open(Path(self.expdir) / "log.log", 'a') as f:
                print(f"{mode} {key}: {value}")
                f.write(f'{mode} at step {global_step}: {value}\n')
                if mode == 'dev' and key == 'roc-auc' and value > self.best_score:
                    self.best_score = torch.ones(1) * value
                    f.write(f'New best {key} on {mode} at step {global_step}: {value}\n')
                    save_names.append(f'{mode}-best.ckpt')

        # if mode in ["dev", "test"]:
        #     with open(Path(self.expdir) / f"{mode}_predict.txt", "w") as file:
        #         lines = [f"{f} {p}\n" for f, p in zip(records["filename"], records["predict_top50"])]
        #         file.writelines(lines)

        #     with open(Path(self.expdir) / f"{mode}_truth.txt", "w") as file:
        #         lines = [f"{f} {l}\n" for f, l in zip(records["filename"], records["truth_top50"])]
        #         file.writelines(lines)

        return save_names