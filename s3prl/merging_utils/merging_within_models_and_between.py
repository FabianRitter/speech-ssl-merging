import os
import glob
import numpy as np
import torch

from time import time
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
from torch import nn

from torchnlp.utils import lengths_to_mask
from graphs.base_graph import NodeType
from metric_calculators import CovarianceMetric, MeanMetric
from matching_functions import match_tensors_permute
from matching_functions import compute_correlation
from merging_utils.ties_merger_adaptation_to_corr import TIESMerging
from inspect import getmembers, isfunction
try:
    import ipdb as pdb
except ImportError:
    import pdb


def contains_name(layer_name, node_list):
    for node in node_list:
        if node in layer_name:
            return True
    return False


def get_merging_fn(name):
    """ Get alignment function from name. """
    import matching_functions
    matching_fns = dict([(k, v) for (k, v) in getmembers(matching_functions, isfunction) if 'match_tensors' in k])
    return matching_fns[name]



class MergeHandler:
    def __init__(self, graph, merge, unmerge, orig):
        self.graph = graph
        # just store merge and unmerge matrices
        self.orig = orig
        self.merge = merge
        self.unmerge = unmerge
        # Define handlers for different module types.
        # self.module_handlers = {
        #     'Conv1d': self.handle_conv1d,
        #     'Conv2d': self.handle_conv2d,
        #     'Linear': self.handle_linear,
        #     'LayerNorm': self.handle_layernorm,
        #     # add others as needed:
        #     'BatchNorm2d': self.handle_batchnorm2d,
        #     # for activation functions that are parameterless:
        #     'GELU': self.handle_fn,
        #     'ReLU': self.handle_fn,
        #     'Tanh': self.handle_fn,
        #     # fallback handler:
        #     'default': self.handle_default
        # }

class ModelMerge(nn.Module):

    def __init__(self, *graphs, device=0):
        super().__init__()
        
        self.hooks = []
        self.init(graphs, device)

    def init(self, graphs, device):

        # move all graph models to eval
        for g in graphs:
            g.model.to(device).eval()

        self.graphs = graphs
        self.device = device
        self.merged_model = None
        count = 0
        for graph in self.graphs:
            print(count)
            count+=1
            graph.add_hooks(device=device)

    # helper function to collect hiddens. Do not recommend using for large FractionalDataloader
    def get_hiddens(self, dataloaders):
        data_stores = [defaultdict(lambda: None) for g in self.graphs]

        with torch.no_grad():
            for dataloader in dataloaders:
                for x, _ in tqdm(dataloader, desc="Forward Pass to Compute Merge Metrics: "):
                    x = x.to(self.device)
                    intermediates = [g.compute_intermediates(x) for g in self.graphs] # shape [feat_dim, num_tokens]
                    nodes = list(intermediates[0].keys())
                    for node in nodes:
                        intermeds_float = [i[node][:,1:-1].float().detach() for i in intermediates] # len = num_graphs
                        if data_stores[0][node] == None:
                            for i in range(len(self.graphs)):
                                data_stores[i][node] = intermeds_float[i]
                        else:
                            for i in range(len(self.graphs)):
                                data_stores[i][node] = torch.cat((data_stores[i][node], intermeds_float[i]), 1)
        return data_stores

    # get average variance across features for each node
    def compute_variances(self, dataloaders):
        data_stores = self.get_hiddens(dataloaders)
        nodes = list(data_stores[0].keys())
                
        for node in nodes:
            for i in range(len(self.graphs)):
                data_stores[i][node] = torch.mean(torch.var(data_stores[i][node], dim=1))
        return data_stores

    # for investigating representations b/w two models 
    def compute_rep_distances(self, dataloaders):
        node_dists = []
        data_stores = self.get_hiddens(dataloaders)
        nodes = list(data_stores[0].keys())

        for node in nodes:
            x = data_stores[0][node]  # shape [feat_dim, num_tokens]
            y = data_stores[1][node]  # shape [feat_dim, num_tokens]
            dists = (x - y).pow(2).sum(0).sqrt()
            node_dists.append(torch.mean(dists))
        return node_dists

    def remove_pads(self, intermediates, input, lens):
        """
        Removes padding from HuBERT intermediates based on downsampled feature lengths.

        Args:
            intermediates: List of feature maps or node representations.
            input: Tensor input to the Transformer.
            lens: List of downsampled lengths for each sequence in the batch.

        Returns:
            intermediates: Updated intermediates with padding removed.
        """
        bsz = len(input)  # Batch size
        
        
        for g_idx in range(len(self.graphs)):
            for node in list(intermediates[0].keys()):
                tensor_to_edit = intermediates[g_idx][node]  # shape: [feat_dim, total_features]
                # Split concatenated tensor based on `lens`
                list_of_tensors = []
                start_idx = 0
                for i in range(bsz):
                    end_idx = start_idx + lens[i]
                    list_of_tensors.append(tensor_to_edit[:, start_idx:end_idx])
                    start_idx = end_idx

                # Concatenate valid features back
                new_tensor = torch.cat(list_of_tensors, dim=1)
                intermediates[g_idx][node] = new_tensor

        return intermediates


    def load_toks(self, saved_path):
        filenames = glob.glob(os.path.join(saved_path, 'toks', '*.pt'))
        num_files = len(filenames)
        tok_ids_all = []
        for i in range(num_files):
            tok_ids_all.append(torch.load(f'{saved_path}/toks/{i}.pt'))
        return torch.cat(tok_ids_all)

    def sent_rep(self, intermediates, node, sentence_level, lens, special_toks=False):
        # already shape [feat_dim, bsz]
        if sentence_level == 'cls':
            return [intermediates[i][node] for i in range(len(self.graphs))]
        bsz = len(lens)
        intermeds_float = []

        if intermediates[0][node].shape[-1] == len(lens): #bsz
            intermeds_float = [intermediates[0][node], intermediates[1][node]]
            return intermeds_float
        for g_idx in range(len(self.graphs)):
            sent_levels = []
            last_idx = 0
            for senlen in lens:
                actual_len = senlen 
                if special_toks == False:
                    actual_len = senlen - 2
                sent_levels.append(intermediates[g_idx][node][:,last_idx:last_idx + actual_len])
                last_idx += actual_len
            if sentence_level == 'maxpool':
                try:
                    sent_avgs = [torch.amax(sent_levels[i].float(), 1).unsqueeze(1) for i in range(bsz)]
                except:
                    breakpoint()
            elif sentence_level == 'avgpool':
                sent_avgs = [torch.mean(sent_levels[i].float(), 1).unsqueeze(1) for i in range(bsz)]
            intermeds_float.append(torch.hstack(sent_avgs)) # list of [[dim, bsz], [dim, bsz]]
        return intermeds_float

    def compute_metrics(self, dataloader, metric_classes, sentence_level=None, special_toks=False, 
                        print_featnorms=False):
        
        self.metrics = None

        if not isinstance(dataloader, list):
            dataloader_list = [dataloader]
        else:
            dataloader_list = dataloader
        

        numel = 0
        downsampling_rate = self.graphs[0].get_downsample_rates(key="encoder")

        for dataloader in dataloader_list:
            for wavs, *others in tqdm(dataloader, desc="Forward Pass to Compute Merge Metrics: "):

                # load batch & track number of elements
                wavs = [torch.FloatTensor(wav).to("cuda") for wav in wavs]
                lens = [wav.shape[-1] // downsampling_rate for wav in wavs]  # Feature lengths per batch

                numel_local =  sum(lens)
                numel += numel_local
                attn_mask = lengths_to_mask(lens)  # Use existing lengths_to_mask function
                # get intermediates and remove padding idxs 
                intermediates = [g.compute_intermediates(wavs) for g in self.graphs]  # shape [feat_dim, num_tokens]
                
                intermediates = self.remove_pads(intermediates, wavs, lens)

                #intermediates = [g.compute_intermediates(wavs) for g in self.graphs] # shape [feat_dim, num_tokens]
                nodes = list(intermediates[0].keys())


                # if qk flag is on, add qk node placeholders for each layer
                qk_flag = False
                # if self.graphs[0].qk == True:
                #     for i in range(self.graphs[0].num_layers):
                #         nodes.append(f'qk{i}')
                #     qk_flag = True
                # populate metrics list 
                if self.metrics is None:
                    self.metrics = {n: {k: v() for k, v in metric_classes.items()} for n in nodes}
                
                # special cases nodes (just q, k)
                special_cases_names = ['q', 'k']
                special_cases_nodes = [self.graphs[0].modules[name] for name in special_cases_names]
                qk_nodes = [self.graphs[0].modules[name] for name in ['q', 'k']]
                
                for node, node_metrics in self.metrics.items():
                    if isinstance(node, int):
                        prev_node_layer = self.graphs[0].get_node_info(node-1)['layer']
                        if prev_node_layer == None or not contains_name(prev_node_layer,special_cases_nodes):
                            for metric in node_metrics.values():
                                if sentence_level != None:
                                    intermeds_float = self.sent_rep(intermediates, node, sentence_level, lens, special_toks)
                                else:
                                    intermeds_float = [i[node].float().detach() for i in intermediates] # len = num_graphs
                                metric.update(len(wavs) , *intermeds_float) 
                        elif contains_name(prev_node_layer, qk_nodes):
                            layer_no = [int(i) for i in self.graphs[0].get_node_info(node-1)['layer'].split('.') if i.isdigit()][0]
                            if qk_flag:
                                qk_metric = self.metrics[f'qk{layer_no}']
                            else:
                                qk_metric = node_metrics
                            for metric in qk_metric.values():
                                if sentence_level != None:
                                    intermeds_float = self.sent_rep(intermediates, node, sentence_level, lens, special_toks)
                                else:
                                    intermeds_float = [i[node].float().detach() for i in intermediates] # len = num_graphs
                                metric.update(len(wavs), *intermeds_float)

        for node, node_metrics in self.metrics.items():
            if isinstance(node, int):
                prev_node_layer = self.graphs[0].get_node_info(node-1)['layer']
                if prev_node_layer == None or not contains_name(prev_node_layer,special_cases_nodes):
                    for metric_name, metric in node_metrics.items():
                        self.metrics[node][metric_name] = metric.finalize(numel, print_featnorms=print_featnorms)
        # if self.graphs[0].qk == True:
        #     for i in range(self.graphs[0].num_layers):
        #         for metric_name, metric in self.metrics[f'qk{i}'].items():
        #             self.metrics[f'qk{i}'][metric_name] = metric.finalize(numel * 2, print_featnorms=print_featnorms) 
        
        return self.metrics, None
                

    def save_features(self, dataloader, sentence_level=False, special_toks=False, 
                        save_feats=False, save_dir=None):
    
        self.metrics = None
        if not isinstance(dataloader, list):
            dataloader_list = [dataloader]
        else:
            dataloader_list = dataloader
        
        numel = 0
        if save_feats:
            tok_indices = []
            feats = [defaultdict(list),defaultdict(list)]

        for dataloader in dataloader_list:
            batch_count = 0
            for x, lens in tqdm(dataloader, desc="Forward Pass to Compute Merge Metrics: "):

                # load batch & track element numbers 
                x = x.to(self.device)
                if sentence_level != None:
                    numel_local = x.shape[0]
                else:
                    numel_local =  sum(lens)
                    if special_toks == False:
                        numel_local =- 2*x.shape[0] # num tokens - BOS/EOS toks 
                numel += numel_local
                    
                # get intermediates and remove padding idxs 
                if 'Bert' in type(self.graphs[0].model).__name__:
                    attn_mask = lengths_to_mask(list(lens))  
                    intermediates =  [g.compute_intermediates(x, attn_mask=attn_mask.long().to(self.device)) for g in self.graphs] # shape [feat_dim, num_tokens]
                else:
                    intermediates = [g.compute_intermediates(x) for g in self.graphs] # shape [feat_dim, num_tokens]
                intermediates = self.remove_pads(intermediates, x, lens, sentence_level, special_toks)

                # store intermediates 
                nodes = list(intermediates[0].keys())
                if save_feats:
                    batch_tok_indices = x.flatten()[torch.argwhere(x.flatten() != 0)].squeeze().detach().cpu()
                    tok_indices.append(batch_tok_indices)
                    for node in nodes:
                        feats[0][node].append(intermediates[0][node].detach().cpu())
                        feats[1][node].append(intermediates[1][node].detach().cpu())

                    # if big enough accumulation, save to file
                    batch_count += 1
                    if batch_count % 1000 == 0:
                        num = batch_count // 1000
                        if batch_count * 8 > 100000:
                            return  
                        print(f'saving features {num}')
                        with open(f'{save_dir}/toks/{num}.pt', 'wb+') as tok_out:
                            torch.save(torch.cat(tok_indices), tok_out) #write toks
                            tok_indices = [] # release memory
                        for model_no in [0, 1]:
                            with open(f'{save_dir}/feats_{model_no}/{num}.pt', 'wb+') as model_out:
                                for node in feats[model_no].keys():
                                    feats[model_no][node] = torch.cat(feats[model_no][node], dim=1)
                                torch.save(feats[model_no], model_out) #write feats
                                feats[model_no] = defaultdict(list) # release memory
                
            if save_feats:
                print('saving last batch')
                num = batch_count // 1000 + 1
                for model_no in [0, 1]:
                    with open(f'{save_dir}/feats_{model_no}/{num}.pt', 'wb+') as model_out:
                        for node in feats[model_no].keys():
                            feats[model_no][node] = torch.cat(feats[model_no][node], dim=1)
                        torch.save(feats[model_no], model_out)
                        feats[model_no] = defaultdict(list)
                with open(f'{save_dir}/toks/{num}.pt', 'wb+') as toks_out:
                    tok_indices = torch.cat(tok_indices)
                    torch.save(tok_indices, toks_out)
                print('finished saving features to file')
        return None, None
    
    ### HELPER FUNCTIONS FOR CORRELATIONS ###

    def compute_np_corr(self, X,Y):
        feats_concat = torch.cat((X.to('cpu'),Y.to('cpu'))).type(torch.float32)
        corr = np.corrcoef(feats_concat)
        corr = np.nan_to_num(corr)
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1)
        return corr

    def compute_np_cov(self, X, Y):
        feats_concat = torch.cat((X.to('cpu'),Y.to('cpu'))).type(torch.float32)
        feats_concat =  feats_concat - feats_concat.mean(dim=1)[:,None]
        cov = (feats_concat @ feats_concat.T).div(feats_concat.shape[1])
        return cov
        
    def cov_to_corr(self, cov, no_corr=False):
        if no_corr == True:
            return cov 
        std = torch.diagonal(cov).sqrt()
        corr = cov / (torch.clamp(torch.nan_to_num(torch.outer(std, std)),min=1e-7))
        return corr

    def separate_res_nodes(self, nodes):
        resnodes = []
        non_resnodes = []
        for node in nodes:
            if self.graphs[0].get_node_info(node)['type'] == NodeType.POSTFIX:
                prev_node_info = self.graphs[0].get_node_info(node-1)['layer']
                if ((self.graphs[0].modules['q'] in prev_node_info) or 
                                            (self.graphs[0].modules['k'] in prev_node_info)):
                    #non_resnodes.append(node) # this is a qk node
                    continue
                else:
                    resnodes.append(node) # all res keys are postfixes by design
            else:
                non_resnodes.append(node)
        return resnodes, non_resnodes


    # load certain number of saved feats
    def load_features(self, saved_path, num, res='first', total_num=10):
        filenames = glob.glob(os.path.join(saved_path, f'feats_{num}', '*.pt'))
        filenames = filenames[:total_num]
        feats_final = {}

        print('loading feats')
        for filename in tqdm(filenames):
            try:
                feats = torch.load(filename)
            except RuntimeError:
                continue
            
            # sort nodes by res or non-res
            resnodes, non_resnodes = self.separate_res_nodes(list(feats.keys()))

            # keep resnodes of interest only
            if res == 'first':
                res_keys_used = [resnodes[0]]
            elif res == 'last':
                res_keys_used = [resnodes[-1]]
            elif res == 'all':
                res_keys_used = resnodes 
            elif res == 'sep':
                res_keys_used = resnodes
            elif res == 'none':
                res_keys_used = []

            # go through non resnodes, and get features ready
            for node in non_resnodes:
                if node not in feats_final:
                    feats_final[node] = feats[node]
                else:
                    feats_final[node] = torch.cat([feats_final[node], feats[node]], dim=1)

            for node in res_keys_used:
                if node not in feats_final:
                    feats_final[node] = feats[node]
                else:
                    feats_final[node] = torch.cat([feats_final[node], feats[node]], dim=1)
                 
            for key in resnodes:
                if key not in feats_final:
                    feats_final[key] = []
        return feats_final

    def compute_corrs(self, nodes, feats_0, feats_1, res='first'):
        corrs = {}

        resnodes, non_resnodes = self.separate_res_nodes(nodes)

        for node in tqdm(non_resnodes):
            if feats_0[node] != []:
                corrs[node] = torch.Tensor(self.compute_np_corr(feats_0[node], feats_1[node]))    
            
        if res == 'first':
            resnode = resnodes[0]
            corrs['res'] = torch.Tensor(self.compute_np_corr(feats_0[resnode], feats_1[resnode]))    
        elif res == 'last':
            resnode = resnodes[-1]
            corrs['res'] = torch.Tensor(self.compute_np_corr(feats_0[resnode], feats_1[resnode]))    
        elif res == 'all':
            node = resnodes[0]
            cov = torch.Tensor(self.compute_np_cov(feats_0[node], feats_1[node]))    
            for node in resnodes[1:]:
                cov += torch.Tensor(self.compute_np_cov(feats_0[node], feats_1[node]))    
            cov /= len(resnodes)
            corrs['res'] = torch.Tensor(self.cov_to_corr(cov))
        elif res == 'sep':
            for node in resnodes:
                corrs[node] = torch.Tensor(self.compute_np_corr(feats_0[node], feats_1[node]))
        # not handling 'none' case for now

        return corrs

    def compute_metric_corrs(self, nodes, res='first', no_corr=False, qk=False):
        corrs = {}
        resnodes, non_resnodes = self.separate_res_nodes(nodes)
        
        for node in tqdm(non_resnodes):
            corrs[node] = self.cov_to_corr(self.metrics[node]['covariance'], no_corr)
        
        if resnodes == []:
            return corrs
        if res == 'first':
            resnode = resnodes[0]
            corrs['res'] = self.cov_to_corr(self.metrics[resnode]['covariance'], no_corr=no_corr)
        elif res == 'last':
            resnode = resnodes[-1]
            corrs['res'] = self.cov_to_corr(self.metrics[resnode]['covariance'], no_corr=no_corr)
        elif res == 'all':
            node = resnodes[0]
            cov = self.metrics[node]['covariance']
            for node in resnodes[1:]:
                cov += self.metrics[node]['covariance']
            cov /= len(resnodes)
            corrs['res'] =self.cov_to_corr(cov, no_corr=no_corr)
        elif res == 'sep':
            for node in resnodes:
                corrs[node] = self.cov_to_corr(self.metrics[node]['covariance'], no_corr=no_corr)
        
        return corrs

    ### END HELPER FUNCTIONS FOR CORRELATIONS ###


    def compute_transformations(self, transform_fn, reduce_ratio=.5, permute_heads=False, 
                                ignore_heads=False, print_costs=False, no_absval=False,
                                saved_features=None, res='first',
                                no_corr=False,**kwargs):

        start_time = time()
        self.merges = {}
        self.unmerges = {}
           

        global_res_merge= None
        global_res_unmerge = None


        special_cases_names = ['final_ln', 'attn_ln', 'q', 'k']
        special_cases_nodes = [self.graphs[0].modules[name] for name in special_cases_names]
        qk_nodes = [self.graphs[0].modules[name] for name in ['q', 'k']]

        cost_dict = {}
        
        if saved_features:
            feats_0 = self.load_features(saved_features, 0, res=res)
            feats_1 = self.load_features(saved_features, 1, res=res)
            nodes = list(feats_0.keys())
            nodes.sort()
            print('computing corrs')
            corrs = self.compute_corrs(nodes, feats_0, feats_1, res=res)
        else:
            nodes = list(self.metrics.keys())
            qk_flag = False
            nodes.sort()
            print('computing corrs')
            corrs = self.compute_metric_corrs(nodes, res=res, no_corr=no_corr, qk=qk_flag)

            # save all corrs to file to look at them. 
            # breakpoint()
            # with open(f'corrs.pt', 'wb+') as corrs_out:
            #     torch.save(corrs, corrs_out)
        
        # corrs has all nonres nodes & the one res node. Unless this is sep, then it has all nodes
        last_node = nodes[-1]
        for node in tqdm(nodes, desc="Computing transformations: "):
            prev_node_layer = self.graphs[0].get_node_info(node-1)['layer']
            # skip metrics associated with residuals and qk if qk is true
            correlation_matrix = None
            if prev_node_layer == None or not contains_name(prev_node_layer,special_cases_nodes):
                if node in corrs:
                    correlation_matrix = corrs[node]

                info = self.graphs[0].get_node_info(node)
                print(info)
                next_node_info = self.graphs[0].get_node_info(node+1)['layer']

                # Handle Attention Merging
                if next_node_info != None and (self.graphs[0].modules['lin_attn'] in next_node_info):
                    layer_no = [int(i) for i in self.graphs[0].get_node_info(node+1)['layer'].split('.') if i.isdigit()][0]
                    if transform_fn.__name__ in ['match_tensors_permute'] and ignore_heads == False:
                        n_heads = self.graphs[0].num_heads
                        mha_transform_fn = transform_fn.__name__ + '_MHA' 
                        merge, unmerge, attn_head_perm, cost = get_merging_fn(mha_transform_fn)(n_heads, r=reduce_ratio, 
                                                                                            permute_heads=permute_heads, print_costs=print_costs, 
                                                                                            no_absval=no_absval, correlation_matrix=correlation_matrix, 
                                                                                            **kwargs)
                        merge = merge * len(self.graphs) 
                        self.merges[node] = merge.chunk(len(self.graphs), dim=1)
                        self.unmerges[node] = unmerge.chunk(len(self.graphs), dim=0)
                        if qk_flag == True:
                            metric = self.metrics[f'qk{layer_no}']
                            correlation_matrix = self.cov_to_corr(metric['covariance'])
                            qk_merge, qk_unmerge, _, cost = get_merging_fn(mha_transform_fn)(n_heads, r=reduce_ratio, 
                                                                                    permute_heads=permute_heads, head_assignments=attn_head_perm, 
                                                                                    print_costs=print_costs, no_absval=no_absval, 
                                                                                    correlation_matrix=correlation_matrix, **kwargs)
                            qk_merge = qk_merge * len(self.graphs)
                            self.merges[f'qk{layer_no}']  = qk_merge.chunk(len(self.graphs), dim=1)
                            self.unmerges[f'qk{layer_no}'] = qk_unmerge.chunk(len(self.graphs), dim=0)
                    else:
                        # if ignoring heads or non-mha merge matrix
                        merge, unmerge, _, cost = transform_fn(reduce_ratio, correlation_matrix=correlation_matrix, 
                                                        no_absval=no_absval, **kwargs)
                        merge = merge * len(self.graphs) 
                        self.merges[node] = merge.chunk(len(self.graphs), dim=1)
                        self.unmerges[node] = unmerge.chunk(len(self.graphs), dim=0)

                        if qk_flag:
                            metric = self.metrics[f'qk{layer_no}']
                            qk_merge, qk_unmerge, _, cost = transform_fn(reduce_ratio, print_costs=print_costs, no_absval=no_absval, 
                                                                    correlation_matrix=correlation_matrix, **kwargs)
                            # add qk_merges to dict here so that attn merge can get added at end of block
                            qk_merge = qk_merge * len(self.graphs)
                            self.merges[f'qk{layer_no}']  = qk_merge.chunk(len(self.graphs), dim=1)
                            self.unmerges[f'qk{layer_no}'] = qk_unmerge.chunk(len(self.graphs), dim=0)
                    
                # Handle FF
                else:
                    # returns merge and unmerge matrixs
                    merge, unmerge, _, cost = transform_fn(reduce_ratio, print_costs=print_costs, no_absval=no_absval, 
                                                        correlation_matrix=correlation_matrix,**kwargs)
                    merge = merge * len(self.graphs) 
                    self.merges[node] = merge.chunk(len(self.graphs), dim=1)
                    self.unmerges[node] = unmerge.chunk(len(self.graphs), dim=0)
            
            elif contains_name(prev_node_layer, qk_nodes):
                continue
                # continuing because this is already handled in attention block

            # handle metrics associated with residuals here, other special cases
            else:
                info = self.graphs[0].get_node_info(node)
                print('res')
                print(info)
                if res == 'sep':
                    correlation_matrix = corrs[node]
                    merge, unmerge, _, cost = transform_fn(reduce_ratio, correlation_matrix=correlation_matrix, 
                                                    no_absval=no_absval,**kwargs)
                    merge = merge * len(self.graphs)
                    self.merges[node] = merge.chunk(len(self.graphs), dim=1)
                    self.unmerges[node] = unmerge.chunk(len(self.graphs), dim=0)
                else:
                    # res is first, last, or all:
                    if global_res_merge == None:
                        correlation_matrix = corrs['res']
                        global_res_merge, global_res_unmerge, _, cost = transform_fn(reduce_ratio,  
                                                                           correlation_matrix=correlation_matrix, 
                                                                           no_absval=no_absval, **kwargs)
                        global_res_merge = global_res_merge * len(self.graphs)
                        self.merges[node] = global_res_merge.chunk(len(self.graphs), dim=1)
                        self.unmerges[node] = global_res_unmerge.chunk(len(self.graphs), dim=0)
                    else: # merge was already learned
                        self.merges[node] = global_res_merge.chunk(len(self.graphs), dim=1)
                        self.unmerges[node] = global_res_unmerge.chunk(len(self.graphs), dim=0)
            cost_dict[node] = cost
        if qk_flag == True:
            for node in nodes:
                prev_node_layer = self.graphs[0].get_node_info(node-1)['layer']
                if prev_node_layer != None and contains_name(prev_node_layer, qk_nodes):
                    layer_no =  [int(i) for i in self.graphs[0].get_node_info(node-1)['layer'].split('.') if i.isdigit()][0]
                    self.merges[node] = self.merges[f'qk{layer_no}']
                    self.unmerges[node] = self.unmerges[f'qk{layer_no}']
            for i in range(self.graphs[0].num_layers):
                self.merges.pop(f'qk{i}')
                self.unmerges.pop(f'qk{i}')
                
        self.compute_transform_time = time() - start_time
        return self.merges, self.unmerges, cost_dict
    
    def find_previous_weighted_node(self, node, graph):
        """Find the nearest predecessor node with trainable parameters (e.g., Conv1d, GroupNorm with affine=True)."""
        current = node
        while True:
            preds = graph.preds(current)
            if not preds:
                return None
            current = preds[0]  # Assuming single predecessor for simplicity
            info = graph.get_node_info(current)
            if info['type'] == NodeType.MODULE:
                module = graph.get_module(info['layer'])
                if isinstance(module, (nn.Conv1d, nn.Linear)) or (isinstance(module, nn.GroupNorm) and module.affine):
                    return preds

        
    def find_next_parameterized_successor(self, node, graph):
        current = node
        while True:
            succs = graph.succs(current)
            if not succs:
                return None
            current = succs[0]  # Assuming single successor
            info = graph.get_node_info(current)
            module = graph.get_module(info['layer'])
            if isinstance(module, (nn.Conv1d, nn.Linear, nn.GroupNorm)) and hasattr(module, 'weight'):
                return succs[0]

    def merge_node(self, node, merger):
        info = merger.graph.get_node_info(node)
        module = merger.graph.get_module(info['layer'])
        
        if isinstance(module, (nn.Linear)): # Handle Linear and Conv layers
            module.weight.data = merger.merge @ module.weight.data
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data = merger.merge @ module.bias.data

        elif isinstance(module, nn.Conv1d):
            module.weight.data = torch.einsum('UO,OIK->UIK', merger.merge, module.weight.data)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data = merger.merge @ module.bias.data
        
        elif isinstance(module, nn.Conv2d):
            module.weight.data = torch.einsum('UO,OIHW->UIHW', merger.merge, module.weight.data)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data = merger.merge @ module.bias.data

        elif isinstance(module, nn.LayerNorm): # Handle LayerNorm
            module.weight.data = merger.merge @ module.weight.data
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data = merger.merge @ module.bias.data
        elif isinstance(module, nn.GroupNorm):  # Handles Fp32GroupNorm, this may be wrong and I may need to do as in nn.Conv1D!. Please revise.!.
            print(f"inside merge_node function for GroupNorm, before if statement.")
            if module.affine:  # Only merge if it has learnable parameters
                print(f"inside merge_node function for GroupNorm, inside if statement.")
                module.weight.data = merger.merge @ module.weight.data
                module.bias.data = merger.merge @ module.bias.data
        else:
            print(f"Warning: merge_node not implemented for layer type: {type(module)} at node {node}")


    def unmerge_node(self, node, merger):
        info = merger.graph.get_node_info(node)
        module = merger.graph.get_module(info['layer'])
        print(f"unmerge_node . unmerging at {module}")
        if isinstance(module, nn.Linear):
            module.weight.data = module.weight @ merger.unmerge
        elif isinstance(module, nn.Conv1d):
            module.weight.data = torch.einsum('OIK,IU->OUK', module.weight.data, merger.unmerge)
        elif isinstance(module, nn.Conv2d):
            module.weight.data = torch.einsum('OIHW,IU->OUHW', module.weight.data, merger.unmerge)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.weight.data = module.weight @ merger.unmerge
        else:
            # Propagate to next parameterized successor
            print(f"no merging at node {node}, Propagate to next parameterized successor")
            succ = self.find_next_parameterized_successor(node, merger.graph)
            if succ:
                self.unmerge_node(succ, merger)
            else:
                print(f"Warning: No parameterized successor found for unmerge at {node}")

        
    # adding custom transformations here, for more control
    def apply_transformations_custom(self, merge_cls=False):
        qk_flag = False
        qk_nodes = [self.graphs[0].modules[name] for name in ['q', 'k']]

        # emb_suff_0 = self.graphs[0].modules['emb']
        # emb_copy_0 = self.graphs[0].get_module(f'{self.graphs[0].enc_prefix}.{emb_suff_0}').weight.data
        # emb_copy_0 = torch.clone(emb_copy_0)

        # emb_suff_1 = self.graphs[1].modules['emb']
        # emb_copy_1= self.graphs[1].get_module(f'{self.graphs[1].enc_prefix}.{emb_suff_1}').weight.data
        # emb_copy_1 = torch.clone(emb_copy_1)
        
        final_merger = None
        graph_device = "cuda"

        for node in self.merges:
            merges = self.merges[node]
            unmerges = self.unmerges[node]
            count = 0
            for merge, unmerge, graph in zip(merges, unmerges, self.graphs):
                merger = MergeHandler(graph, merge, unmerge, node)
                merger.merge = merger.merge.to(graph_device)
                merger.unmerge = merger.unmerge.to(graph_device)

                #preds = merger.graph.preds(node)
                preds = self.find_previous_weighted_node(node, graph)
                info = merger.graph.get_node_info(preds[0])
                module = merger.graph.get_module(info['layer']) # Getting module based on predecessor node's layer name


                if isinstance(module, nn.Conv1d): # Merging Conv1D layers
                    print(f'merging Conv1D: {info["layer"]}')
                    self.merge_node(preds[0], merger)
                    #succ = merger.graph.succs(node)[0] # Assuming one successor after Conv1D for unmerge
                    succ = self.find_next_parameterized_successor(node, merger.graph)
                    if succ:
                        self.unmerge_node(succ, merger)
                elif isinstance(module, nn.GroupNorm):  # Handles Fp32GroupNorm too
                    print(f'merging GroupNorm: {info["layer"]}')
                    self.merge_node(preds[0], merger)
                    #succ = merger.graph.succs(node)[0]
                    succ = self.find_next_parameterized_successor(node, merger.graph)
                    if succ:
                        self.unmerge_node(succ, merger)
                    #if hasattr(succ_module, 'weight'):  # Unmerge only if successor has parameters


                # self attention merging, and self attention out unmerging 
                elif info['type'] == NodeType.SUM:
                    print('merging MHA')
                    # apply merges to k,q,v matrices
                    sum_preds = merger.graph.preds(preds[0])
                    # check if q,k junction or v matrix
                    for sum_pred in sum_preds:
                        info = merger.graph.get_node_info(sum_pred)
                        if info['type'] == NodeType.SUM:
                            if qk_flag == False:
                                second_sum_preds = merger.graph.preds(sum_pred)
                                # merge q & k 
                                for second_sum_pred in second_sum_preds:
                                    self.merge_node(second_sum_pred, merger)
                        elif 'v_proj' in info['layer'] or 'value' in info['layer']:
                            # merge v
                            self.merge_node(sum_pred, merger)

                    # unmerge self-attn.out
                    succ = merger.graph.succs(node)[0]
                    self.unmerge_node(succ, merger)
                elif contains_name(info['layer'], qk_nodes) and qk_flag == True:
                    print('merging qk')
                    self.merge_node(preds[0], merger)

                elif 'self_attn_layer_norm' in info['layer'] or 'attention.output.LayerNorm' in info['layer']:
                    raise NotImplementedError(
                        "self-attn layer-norm residual merging is not supported in this code path "
                        f"(layer: {info['layer']}); use model_merger_new.ModelMerge instead.")
                    print('merging self-attn res')
                    # apply merge to ln
                    module = merger.graph.get_module(info['layer'])
                    parameter_names = ['weight', 'bias']
                    for parameter_name in parameter_names:
                        parameter = getattr(module, parameter_name)
                        parameter.data = merger.merge @ parameter

                    # apply merges to the self.attn out proj
                    sum = merger.graph.preds(preds[0])[0]
                    out_proj = merger.graph.preds(sum)[0]
                    self.merge_node(out_proj, merger)

                    # unmerge the ff1 module 
                    ff1 = merger.graph.succs(node)[0]
                    self.unmerge_node(ff1, merger)

                elif 'final_layer_norm' in info['layer'] or 'layernorm_embedding' in info['layer'] or 'output.LayerNorm' in info['layer'] or 'embeddings.LayerNorm' in info['layer']:
                    print('merging final res')
                    # apply merge to ln
                    module = merger.graph.get_module(info['layer'])
                    parameter_names = ['weight', 'bias']
                    for parameter_name in parameter_names:
                        parameter = getattr(module, parameter_name)
                        parameter.data = merger.merge @ parameter

                    sum = merger.graph.preds(preds[0])[0]
                    info = merger.graph.get_node_info(sum)
                    if info['type'] == NodeType.SUM:
                        ff2 = merger.graph.preds(sum)[0]
                        self.merge_node(ff2, merger)
                    else:
                        # this is emb node then
                        if final_merger == None and count == 1:
                            final_merger = merger
                        if merger.graph.enc_prefix == 'bert':
                            # bert has special token type embedding that must be merged too
                            emb_tok_suff = merger.graph.modules['emb_tok_type']
                            emb_tok_name = f'{merger.graph.enc_prefix}.{emb_tok_suff}'
                            emb_tok_mod = merger.graph.get_module(emb_tok_name)
                            emb_tok_mod.weight.data = (merger.merge @ (emb_tok_mod.weight).T).T 

                        # grabbing naming vars
                        emb_suff = merger.graph.modules['emb']
                        emb_pos_suff = merger.graph.modules['emb_pos']
                        emb_name = f'{merger.graph.enc_prefix}.{emb_suff}'
                        emb_pos_name = f'{merger.graph.enc_prefix}.{emb_pos_suff}'

                        # merger emb &  emb_pos
                        emb = merger.graph.get_module(emb_name)
                        emb_pos = merger.graph.get_module(emb_pos_name)
                        emb.weight.data = (merger.merge @ (emb.weight).T).T
                        emb_pos.weight.data = (merger.merge @ (emb_pos.weight).T).T 

                    # this unmerges w_k, w_q, w_v
                    succs = merger.graph.succs(node)
                    if len(succs) > 1:
                        for succ in succs:
                            info = merger.graph.get_node_info(succ)
                            if info['type'] != NodeType.SUM:
                                self.unmerge_node(succ, merger)
                    else:
                        # in this case, we have the second to last node
                        # separate case for mnli & camembert due to head names
                        # first we check if model is bert and unmerge the lm head 
                        if 'cls.predictions.transform.dense' in merger.graph.named_modules:
                            module = merger.graph.get_module('cls.predictions.transform.dense') 
                            module.weight.data = module.weight @ merger.unmerge

                        elif 'bert.pooler.dense' in merger.graph.named_modules:
                            module = merger.graph.get_module('bert.pooler.dense') 
                            module.weight.data = module.weight @ merger.unmerge
                        elif len(merger.graph.model.classification_heads.keys()) != 0:
                            if 'classification_heads.mnli.dense' in merger.graph.named_modules:
                                module = merger.graph.get_module('classification_heads.mnli.dense')
                                module.weight.data = module.weight @ merger.unmerge
                            elif 'classification_heads.sentence_classification_head.dense' in merger.graph.named_modules:
                                module = merger.graph.get_module('classification_heads.sentence_classification_head.dense')
                                module.weight.data = module.weight @ merger.unmerge
                        # if has no classification heads, it uses lm heads instead, and is a roberta model
                        # unmerge this, but in the actual eval of wsc, need to fix forward pass, but this is the minimum needed to
                        # store the correct weights
                        else:
                            module = merger.graph.get_module('encoder.lm_head.dense')
                            module.weight.data = module.weight @ merger.unmerge

                # apply merge to fc1 & unmerge fc2
                elif 'fc1' in info['layer'] or 'intermediate.dense' in info['layer']:
                    print('merging ff')
                    # apply merges to the fc1 layer
                    module = merger.graph.get_module(info['layer'])
                    self.merge_node(preds[0], merger)
                    
                    # apply unmerge to fc2 layer
                    succ = merger.graph.succs(node)[0]
                    self.unmerge_node(succ, merger)
                
                elif 'transform.LayerNorm' in info['layer'] and merge_cls:
                    if final_merger == None and count == 1: # count ensures this is 2nd model merger being saved
                        final_merger = merger

                    print('merging lm head')
                    # apply merge to layernorm 
                    module = merger.graph.get_module(info['layer'])
                    parameter_names = ['weight', 'bias']
                    for parameter_name in parameter_names:
                        parameter = getattr(module, parameter_name)
                        parameter.data = merger.merge @ parameter
                    
                    # merge dense
                    pred = merger.graph.preds(preds[0])[0]
                    self.merge_node(pred, merger)

                elif 'pooler' in info['layer'] and merge_cls:
                    print('merging class head')
                    # merge pooler weight
                    self.merge_node(preds[0], merger)
                    # get cls node & unmerge
                    succ = merger.graph.succs(node)[0]
                    self.unmerge_node(succ, merger)
                count += 1

        return final_merger
        
    def get_merged_state_dict(self, interp_w=None, save_both=False, use_ties=False, quantile=0.8, maintain_hubert_behavior=True):
        """
        Post transformations, obtain state dictionary for merged model by linearly interpolating between 
        transformed models in each graph. By default all parameters are averaged, but if given an interp_w 
        weight, will be weightedly averaged instead.
        - interp_w (Optional): If None, all parameters of each model is averaged for merge. Otherwise, 
        interp_w is a list of len(num_models_to_merge), with weights bearing the importance of incorporating 
        features from each model into the merged result.
        use_ties (bool): If True, applies TIES-Merging.
        quantile (float): Quantile for trimming in TIES-Merging.
        Returns: state dict of merged model.
        """
        if save_both:
            merged_state_dict1 = self.graphs[0].model.state_dict().copy()
            merged_state_dict2 = self.graphs[1].model.state_dict().copy()
            return [merged_state_dict1, merged_state_dict2]
        else:
            if use_ties:
                hubert_dict = self.graphs[0].model.state_dict()
                mert_dict = self.graphs[1].model.state_dict()
                # Use TIESMerging class
                ties_merger = TIESMerging(hubert_dict, mert_dict, quantile=quantile, interp_w=interp_w, maintain_hubert_behavior=maintain_hubert_behavior)
                state_dict = ties_merger.merge()
            else:
                state_dict = {}
                merged_state_dict = self.merged_model.state_dict()
                keys = list(self.graphs[0].model.state_dict().keys())
                try:
                    for key in keys:
                        if key in merged_state_dict:
                            param = self.graphs[0].model.state_dict()[key]
                            if interp_w is not None and param.shape == merged_state_dict[key].shape:
                                print(f"merging params with interp_w: {interp_w}")
                                new_value = sum(graph.model.state_dict()[key] * w for graph, w in zip(self.graphs, interp_w))
                            else:
                                new_value = sum(graph.model.state_dict()[key] for graph in self.graphs) / len(self.graphs)
                            state_dict[key] = new_value
                except RuntimeError as e:
                    # Only catch runtime errors about tensor sizes, we need to be able to add models with diff heads together
                    if 'size' not in str(e):
                        raise e
            return state_dict
        


    def clear_hooks(self):
        """ Clears all hooks from graphs. """
        for g in self.graphs:
            g.clear_hooks()
        for hook in self.hooks:
            hook.remove()
        self.hooks = []      

              
    def transform(self, model,
                  dataloader,
                  sentence_level=None,
                  special_toks=False,
                  transform_fn=match_tensors_permute,
                  metric_classes=(CovarianceMetric, MeanMetric),
                  save_both=False,
                  permute_heads=False,
                  ignore_heads=False,
                  no_absval=False,
                  merge_cls=False,
                  saved_features=None,
                  res_type='none',
                  merge_cnn=True,  # NEW FLAG but not being used here to be hoinest
                  interp_w=None,
                  **transform_kwargs
                  ):
        """ Note: this consumes the models given to the graphs. Do not modify the models you give this. """
        
        self.merged_model = model.to(self.device).eval() # same arch as graph models
                
        if not isinstance(metric_classes, dict):
            metric_classes = { x.name: x for x in metric_classes }
        
        self.metric_classes = metric_classes
        self.transform_fn = transform_fn

        # if we did not pre-save features, compute them here:
        if saved_features == None:
            _, vars = self.compute_metrics(dataloader, 
                                metric_classes=metric_classes, 
                                sentence_level=sentence_level,
                                special_toks=special_toks)


        _, _, cost_dict = self.compute_transformations(transform_fn, reduce_ratio=1 - 1. / len(self.graphs),
                                    permute_heads=permute_heads,
                                    ignore_heads=ignore_heads,
                                    no_absval=no_absval, 
                                    saved_features=saved_features,
                                    res=res_type,
                                    **transform_kwargs
                                    )

        final_merger = self.apply_transformations_custom(merge_cls=merge_cls)

        if save_both:
            #self.merged_model1 = deepcopy(self.graphs[0].model).to(self.device)
            def load_hubert_base(model_name="hubert_base"):
                # Load HuBERT base model from s3prl
                #os.environ["TORCH_HOME"] = "/workspace/s3prl/s3prl/cache"
                #base_model = torch.hub.load("s3prl/s3prl", model_name).cuda()
                import s3prl.hub as hub
                merge_type = 'ff+attn'  # Change this to experiment with other types
                model = getattr(hub, 'hubert_base')()
                device = 'cuda'  # or cpu
                model = model.to(device)
                model.model.encoder.layerdrop = 0  # Ensure no dropout in encoder layers
                return model

            self.merged_model1 = load_hubert_base(model_name="hubert_base").to(self.device)
            self.merged_model1.load_state_dict(self.graphs[0].model.state_dict())
            #self.merged_model2 = deepcopy(self.graphs[1].model).to(self.device)
            self.merged_model2 = load_hubert_base(model_name="hubert_base").to(self.device)
            self.merged_model2.load_state_dict(self.graphs[1].model.state_dict())

        if save_both:
            merged_dicts = self.get_merged_state_dict(save_both=True, interp_w=interp_w,use_ties=transform_kwargs["use_ties"], quantile=transform_kwargs["quantile"], maintain_hubert_behavior=transform_kwargs["maintain_hubert_behavior"])
            self.merged_model1.load_state_dict(merged_dicts[0])
            self.merged_model2.load_state_dict(merged_dicts[1])
            self.merged_model.load_state_dict(self.get_merged_state_dict(interp_w=interp_w, save_both=False, use_ties=transform_kwargs["use_ties"], quantile=transform_kwargs["quantile"], maintain_hubert_behavior=transform_kwargs["maintain_hubert_behavior"] ), strict=False)
        else:
            self.merged_model.load_state_dict(self.get_merged_state_dict(interp_w=interp_w, save_both=False, use_ties=transform_kwargs["use_ties"], quantile=transform_kwargs["quantile"], maintain_hubert_behavior=transform_kwargs["maintain_hubert_behavior"] ), strict=False)
        self.add_hooks()

        if final_merger == None:
            unmerge = None
        else:
            unmerge = final_merger.unmerge

        return unmerge, cost_dict
    
    def add_hooks(self):
        """ Add hooks at zip start or stop at locations for merged model and base models. """
        # Remove the hooks from the models to add or own
        self.clear_hooks()
        

