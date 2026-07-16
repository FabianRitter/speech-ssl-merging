''''
This file is included from the https://github.com/gstoica27/ZipIt repository.
'''

import torch
import networkx as nx
from enum import Enum
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pdb
from s3prl.upstream.interfaces import UpstreamBase


class FeatureReshapeHandler:
    """ Instructions to reshape layer intermediates for alignment metric computation. """
    def handle_conv2d(self, x):
        # reshapes conv2d representation from [B, C, H, W] to [C, -1]
        B, C, H, W = x.shape
        return x.permute(1, 0, 2, 3).reshape(C, -1)
    def handle_conv1d(self, x):
        # x is expected to have shape [B, C, L]
        B, C, L = x.shape
        # Bring channels to the front and flatten the batch and length dimensions:
        return x.permute(1, 0, 2).reshape(C, -1)

    # def handle_linear(self, x):
    #     # x is shape [seq_len, batch_szd, C]. Want [C, -1]
    #     x = x.flatten(0, len(x.shape)-2).transpose(1, 0).contiguous()
    #     return x
    def handle_linear(self, x):
        # Input x assumed to be [T, B, C] or [B, T, C]
        # Output desired: [C, Tokens] = [C, T*B]
        if x.dim() == 3:
             if x.shape[1] > x.shape[0] and x.shape[1] > x.shape[2]: # Likely (T, B, C)
                  x = x.permute(1, 0, 2) # -> (B, T, C)
             # Now assume (B, T, C)
             x = x.permute(2, 0, 1) # -> (C, B, T)
             x = x.flatten(1, 2) # -> (C, B*T)
        elif x.dim() == 2: # Should not happen for transformer intermediates?
             print(f"Warning: handle_linear received 2D input {x.shape}")
             x = x.t() # Assume (Tokens, C) -> (C, Tokens)
        # Add handling for other dims if necessary
        return x.contiguous()
    # ... (other handlers) ...

    # --- NEW: Handler specifically for the attention output (pre-out_proj) ---
    # It has shape [T, B, E], same as input to Linear layers
    def handle_attention_output(self, x):
        return self.handle_linear(x) # Reuse linear logic

    def handle_qkv_input(self, x):
        # Input shape is the layer input, typically [T, B, E]
        return self.handle_linear(x)
    
    def handle_identity(self, x):
        print(f"handle identity activated.. still on dropout")
        return x # No reshaping, just return input
    
    def handle_transformer_sentence_encoder_layer(self, x):
        """
        Reshapes the tensor output from the TransformerSentenceEncoderLayer.

        The input `x` is expected to be the attention output tensor with the shape:
        [seq_len, batch_size, embedding_dim].

        The goal is to reshape this to [embedding_dim, -1] for alignment purposes.
        """
        # Check that the tensor has the expected dimensionality
        if len(x.shape) != 3:
            raise ValueError(
                f"Expected 3D tensor for TransformerSentenceEncoderLayer, but got shape {x.shape}"
            )

        # Reshape the tensor from [seq_len, batch_size, embedding_dim]
        # to [embedding_dim, seq_len * batch_size]
        # Transpose to [batch_size, seq_len, embedding_dim]
        x = x.transpose(1, 0).contiguous()
        # Reshape to [embedding_dim, -1] by flattening batch_size and seq_len
        x = x.flatten(0, 1).transpose(1, 0).contiguous()
        return x
    
    def __init__(self, class_name, info):
        self.handler = {
            'BatchNorm2d': self.handle_conv2d,
            'LayerNorm': self.handle_linear,
            'Conv2d': self.handle_conv2d,
            'Conv1d': self.handle_conv1d,
            'Linear': self.handle_linear,
            'AttentionOutput': self.handle_attention_output, # For pre-out_proj hook
            'GELU': self.handle_conv1d,
            'AdaptiveAvgPool2d': self.handle_conv2d,
            'LeakyReLU': self.handle_conv2d,
            'ReLU': self.handle_conv2d, 
            'Tanh': self.handle_conv2d,
            'MaxPool2d': self.handle_conv2d,
            'AvgPool2d': self.handle_conv2d,
            'SpaceInterceptor': self.handle_conv2d,
            'QKVInput': self.handle_qkv_input,      # For Q/K/V input captured via callback
            'Identity': self.handle_linear,
            'Dropout': self.handle_conv1d,  # Add this line
            'Fp32GroupNorm': self.handle_conv1d,
            'TransformerSentenceEncoderLayer': self.handle_transformer_sentence_encoder_layer,
            'MultiheadAttention': self.handle_linear, # Output is typically [T, B, C] or [B, T, C]
            'PermuteAndUnpermute': self.handle_linear, # Assume underlying is linear-like

            
        }[class_name]
        self.info = info
    
    def reshape(self, x):
        x = self.handler(x)

        # Handle modules that we only want a piece of
        if self.info['chunk'] is not None:
            idx, num_chunks = self.info['chunk']
            x = x.chunk(num_chunks, dim=0)[idx]

        return x
    

class NodeType(Enum):
    MODULE = 0          # node is torch module
    PREFIX = 1          # node is a PREFIX (i.e., we want to hook inputs to child node)
    POSTFIX = 2         # node is a POSTFIX  (i.e., we want to hook outputs to parent node)
    SUM = 3             # node is a SUM (e.g., point where residual connections are connected - added)
    CONCAT = 4          # node is a CONCATENATION (e.g, point where residual connections are concatenated)
    INPUT = 5           # node is an INPUT (graph starting point)
    OUTPUT = 6          # node is an OUTPUT (graph output point)
    EMBEDDING = 7       # node is an embedding module (these can only be merged)

# --- Hook Factory Functions ---
def make_prehook(graph_instance, node_id, node_info, device, handler_key):
    """Factory to create a pre-hook function that accesses graph_instance.intermediates."""
    def prehook(module, input_tuple):
        # ... (input validation checks) ...
        if not isinstance(input_tuple, tuple) or not input_tuple: return None
        input_tensor = input_tuple[0]
        if not isinstance(input_tensor, torch.Tensor) or input_tensor.numel() == 0: return None
        try:
            # print(f"  DEBUG PREHOOK Triggered: Node {node_id} Input Shape: {input_tensor.shape}")
            handler = FeatureReshapeHandler(handler_key, node_info)
            reshaped_input = handler.reshape(input_tensor.to(device))
            # --- MODIFICATION: Use graph_instance.intermediates ---
            graph_instance.intermediates[node_id] = reshaped_input
            # print(f"  DEBUG PREHOOK Captured: Node {node_id}, Stored in dict id {id(graph_instance.intermediates)}")
        except Exception as e:
            print(f"Error in prehook processing for PREFIX node {node_id} ...: {e}")
            # import traceback; traceback.print_exc() # Uncomment for details
        return None
    return prehook

# Pass 'graph_instance' (which is 'self' from BIGGraph)
def make_posthook(graph_instance, node_id, node_info, device, handler_key):
    """Factory to create a post-hook function that accesses graph_instance.intermediates."""
    def posthook(module, input_tuple, output):
        # ... (output validation checks) ...
        if not isinstance(output, torch.Tensor) or output.numel() == 0: return None
        try:
            # print(f"  DEBUG POSTHOOK Triggered: Node {node_id} Output Shape: {output.shape}")
            handler = FeatureReshapeHandler(handler_key, node_info)
            reshaped_output = handler.reshape(output.to(device))
            # --- MODIFICATION: Use graph_instance.intermediates ---
            graph_instance.intermediates[node_id] = reshaped_output
            # print(f"  DEBUG POSTHOOK Captured: Node {node_id}, Stored in dict id {id(graph_instance.intermediates)}")
        except Exception as e:
            print(f"Error in posthook processing for POSTFIX node {node_id} ...: {e}")
            # import traceback; traceback.print_exc() # Uncomment for details
        return None
    return posthook

def make_mha_posthook(graph_instance, node_id, node_info, device):
    """Post-hook for MultiheadAttention that extracts attn_output from tuple output.
    MHA.forward() returns (attn_output, attn_weights) where attn_output is [T, B, embed_dim].
    PyTorch's MHA uses F.linear() for out_proj internally, so pre-hooks on out_proj don't fire.
    This post-hook on the full MHA module captures the output AFTER out_proj."""
    def posthook(module, input_tuple, output):
        try:
            # MHA output is (attn_output, attn_weights) or just attn_output
            attn_output = output[0] if isinstance(output, tuple) else output
            if not isinstance(attn_output, torch.Tensor) or attn_output.numel() == 0:
                return None
            # attn_output shape: [T, B, embed_dim] — reshape to [embed_dim, T*B]
            T, B, C = attn_output.shape
            reshaped = attn_output.permute(2, 0, 1).reshape(C, T * B).to(device)
            graph_instance.intermediates[node_id] = reshaped
        except Exception as e:
            print(f"Error in MHA posthook for node {node_id}: {e}")
        return None
    return posthook

class BIGGraph(ABC):
    def __init__(self, model):
        """Initialize DAG of computational flow for a model."""
        self.reset_graph()
        self.named_modules = dict(model.named_modules())
        self.named_params = dict(model.named_parameters())
        self.model = model
        self.intermediates = {}
        self.hooks = [] # Stores handles to attached PyTorch hooks
        self._hook_registry = {} # Temporary storage for MHA callbacks during add_hooks
        self.modules = {} # Populated by subclass (e.g., HuBERTGraph)

        # Remove existing hooks if necessary
        if isinstance(self.model, UpstreamBase):
            print("Removing all existing hooks from the model...")
            # Assuming remove_all_hooks exists and works
            if hasattr(self.model, 'remove_all_hooks') and callable(self.model.remove_all_hooks):
                self.model.remove_all_hooks()
            else:
                print("Warning: model does not have a callable 'remove_all_hooks' method.")

        self.working_info = {}
        self.unmerged = set()
        self.merged = set()
        print(f"BIGGraph Initialized for model type: {type(model)}")
    
    def reset_graph(self):
        """ Create New Graph. """
        self.G = nx.DiGraph()
    
    def preds(self, node):
        """ Get predessors from a node (layer). """
        return list(self.G.pred[node])
    
    def succs(self, node):
        """ Get successors from a node (layer). """
        return list(self.G.succ[node])
    
    def get_node_info(self, node_name):
        """ Get attribute dict from node. """
        return self.G.nodes()[node_name]
    
    def get_module_from_node(self, node_name):
        """ Get pytorch module associated with node. """
        info = self.get_node_info(node_name)
        if info['type'] == NodeType.MODULE:
            return self.named_modules[info["layer"]]
        else:
            raise ValueError(f"Tried to get module from {node_name} of type {info['type']}.")
    
    def get_module(self, module_name):
        """ Get module parameters. """
        return self.named_modules[module_name]
    
    def get_parameter(self, param_name):
        """ Get parameter from name. """
        return self.named_params[param_name]
    
    def get_node_str(self, node_name):
        """ Get node type name. """
        info = self.get_node_info(node_name)
        
        if info['type'] == NodeType.MODULE:
            return self.get_module_from_node(node_name).__class__.__name__
        else:
            return info['type'].name
    
    def create_node_name(self):
        """woo magic. A robust id generator """
        return len(self.G)

    def create_node(self,
                    node_name=None,
                    layer_name=None,
                    param_name=None,
                    node_type=NodeType.MODULE,
                    chunk=None,
                    special_merge=None):
        """ 
        Create node to be added to graph. All arguments are optional, but 
        specify different kinds of node properties. 
        Arguments:
        - node_name: unique identifier for a node. If None, a unique id will be generated 
            via complex hashing function.
        - layer_name: name of pytorch module node represents. layer_name MUST match the module name.
        - node_type: type of node created. By default it is a MODULE (pytorch module), but can also 
            commonly be POSTFIX or PREFIX. These latter specify the node to a place where an alignment 
            between models will be computed and applied.
        - chunk: Whether node represents a disjoint part of a module which other nodes also are a part of.
            Chunk is (i, total). 
        - special_merge: Whether to apply a specific merge/unmerge operation on at this node specially. 
        If none, the transform_fn from model_merger will be applied.
        """
        if node_name is None:
            node_name = self.create_node_name()
        self.G.add_nodes_from([(node_name, {
            'layer': layer_name,
            'type': node_type,
            'param': param_name,
            'chunk': chunk,
            'special_merge': special_merge
        })])
        return node_name

    def add_directed_edge(self, source, target, **kwargs):
        """ Add an edge from source node to target node. """
        self.G.add_edge(source, target, **kwargs)

    def add_nodes_from_sequence(self, name_prefix, list_of_names, input_node, sep='.'):
        """ 
        Add multiple nodes in sequence by creating them and adding edges between each. 
        Args: 
        - name_prefix: Least common ancestor module name string all nodes share. 
            Usually this is the name of a nn.Sequential layer
        - list_of_names: list of module names. Can be the ordered module names in an nn.Sequential.
        - input_node: source node the sequence is attached to.
        Returns: 
        - output sequence node.
        """
        source_node = input_node
        for name in list_of_names:
            if isinstance(name, str):
                if name_prefix != '':
                    temp_node = self.create_node(layer_name=name_prefix + f'{sep}{name}')
                else:
                    temp_node = self.create_node(layer_name= f'{name}')
            else:
                temp_node = self.create_node(node_type=name)
            self.add_directed_edge(source_node, temp_node)
            source_node = temp_node
        return source_node
    
    def print_prefix(self):
        """ Print (POST/PRE)FIX node inputs and outputs. """
        for node in self.G:
            info = self.get_node_info(node)
            if info['type'] in (NodeType.PREFIX, NodeType.POSTFIX):
                print(f'{node:3} in={len(self.preds(node))}, out={len(self.succs(node))}')

    def draw(self, nodes=None, save_path=None):
        """
        Visualize DAG. By default all nodes are colored gray, but if parts of module have already been 
        transformed, they will be colored according to the kinds of transformations applied on the nodes.
        Color Rubric: 
        - Gray: Not merged or unmerged  (output space is not aligned        and input space is not  aligned)
        - Blue: merged but not unmerged (output space is     aligned,       but input space is not  aligned)
        - Red: Not merged but unmerged  (output space is not aligned,       but input space is      aligned)
        - Pink: Merged and Unmerged     (output space is     aligned,       and input space is      aligned)
        
        Args:
        - nodes (optional): list of indices of nodes to visualize. If None, all nodes will be drawn.
        - save_path (optional): path in which to save graph. 
        """
        G = self.G
        if nodes is not None:
            G = nx.subgraph(G, list(nodes))
            
        labels = {i: f'[{i}] ' + self.get_node_str(i) for i in G}
        pos = nx.nx_agraph.graphviz_layout(G, prog='neato')
        node_size = [len(labels[i])**2 * 60 for i in G]
        
        colors = {
            (False, False): (180, 181, 184),
            (True, False): (41, 94, 255),
            (False, True): (255, 41, 91),
            (True, True): (223, 41, 255),
        }
        
        for k, v in colors.items():
            colors[k] = tuple(map(lambda x: x / 255., v))
        
        node_color = [colors[(node in self.merged, node in self.unmerged)] for node in G]
        plt.figure(figsize=(120, 160))
        nx.draw_networkx(G, pos=pos, labels=labels, node_size=node_size, node_color=node_color)
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
    
    
    def add_hooks(self, device="cuda"):
        """Attaches PyTorch hooks or registers callbacks for intermediate capture."""
        if not self.modules:
             print("WARNING: self.modules dictionary is empty in BIGGraph. Suffixes may be incorrect.")
             # Define default suffixes here as a fallback if needed
             self.modules = {
                 'q': 'self_attn.q_proj', 'k': 'self_attn.k_proj', 'v': 'self_attn.v_proj',
                 'lin_attn': 'self_attn.out_proj', 'fc1': 'fc1', 'fc2': 'fc2'
             }

        self.clear_hooks() # Start fresh
        print(f"--- Adding hooks for intermediate capture on device {device} ---")

        q_proj_suffix = self.modules.get('q', 'self_attn.q_proj')
        k_proj_suffix = self.modules.get('k', 'self_attn.k_proj')
        v_proj_suffix = self.modules.get('v', 'self_attn.v_proj')
        attn_out_proj_suffix = self.modules.get('lin_attn', 'self_attn.out_proj')
        fc2_suffix = self.modules.get('fc2', 'fc2')
        # Prefix node *after* fc1 implies its layer name might contain fc1
        fc1_layer_name_part = f".{self.modules.get('fc1', 'fc1')}"

        nodes_processed = set() # Keep track to avoid redundant operations if graph has cycles/merges

        for node_id in self.G:
            if node_id in nodes_processed: continue
            info = self.get_node_info(node_id)

            if info['type'] == NodeType.PREFIX:
                # --- Find MODULE Successor (using BFS) ---
                successor_module_node = None
                successor_info = None
                queue = list(self.G.succ[node_id])
                visited = {node_id} | set(queue) # Initialize visited with node and immediate successors
                while queue:
                    succ_node_candidate = queue.pop(0)
                    succ_info_candidate = self.get_node_info(succ_node_candidate)
                    if succ_info_candidate['type'] == NodeType.MODULE:
                        successor_module_node = succ_node_candidate
                        successor_info = succ_info_candidate
                        # print(f"DEBUG Node {node_id}: Found MODULE successor Node {successor_module_node}, Layer: {successor_info.get('layer', 'N/A')}")
                        break
                    for next_succ in self.G.succ[succ_node_candidate]:
                        if next_succ not in visited:
                            visited.add(next_succ)
                            queue.append(next_succ)
                # --- ---

                if successor_info is None:
                    print(f"ERROR: PREFIX node {node_id} ({info.get('layer', 'NoLayer')}) had no subsequent MODULE node.")
                    nodes_processed.add(node_id)
                    continue

                succ_layer_name = successor_info.get('layer', '')
                parent_attn_layer_name = None # Determine if related to MHA
                if any(s in succ_layer_name for s in [q_proj_suffix, k_proj_suffix, v_proj_suffix, attn_out_proj_suffix]):
                     parent_attn_layer_name = succ_layer_name.rsplit('.', 1)[0]

                # --- Strategy 1: Callback Registration for QKV MHA components ---
                # NOTE: attn out_proj uses Strategy 2 (standard pre-hook) instead of callbacks,
                # because the callback mechanism (_graph_hook_registry_ref) is never invoked
                # by TransformerSentenceEncoderLayer.forward(). Standard PyTorch pre-hooks
                # on nn.Linear modules fire automatically and reliably.
                if parent_attn_layer_name and attn_out_proj_suffix not in succ_layer_name:
                    hook_key, target_name, handler_key = None, None, None
                    if q_proj_suffix in succ_layer_name: hook_key, target_name, handler_key = 'q_input_hook', 'Q_input', 'QKVInput'
                    elif k_proj_suffix in succ_layer_name: hook_key, target_name, handler_key = 'k_input_hook', 'K_input', 'QKVInput'
                    elif v_proj_suffix in succ_layer_name: hook_key, target_name, handler_key = 'v_input_hook', 'V_input', 'QKVInput'

                    if hook_key:
                        try:
                            parent_attn_module = self.get_module(parent_attn_layer_name)
                            # Factory to create the actual callback function capturing scope
                            def make_callback(nid, s_info, hkey, tname):
                                def callback_func(tensor):
                                    try:
                                        reshaper = FeatureReshapeHandler(hkey, s_info)
                                        self.intermediates[nid] = reshaper.reshape(tensor.to(device))
                                    except Exception as e: print(f"ERROR in callback {tname}..: {e}")
                                return callback_func

                            # Register the callback created by the factory
                            callback = make_callback(node_id, successor_info, handler_key, target_name)
                            self._register_hook_callback(parent_attn_module, hook_key, callback)
                        except KeyError: print(f"ERROR: Module {parent_attn_layer_name} not found for MHA callback registration (Node {node_id})")
                        except Exception as e: print(f"ERROR registering MHA callback for node {node_id}: {e}")

                # --- Strategy 2: Standard Pre-Hook for non-QKV components ---
                # This handles: fc2 (FFN), conv_layers (CNN), AND attn out_proj (attention)
                else:
                    handler_key = None
                    use_mha_posthook = False

                    if attn_out_proj_suffix in succ_layer_name:
                        # MHA out_proj uses F.linear() internally — pre-hooks don't fire.
                        # Use a post-hook on the parent self_attn module instead.
                        use_mha_posthook = True
                    elif fc2_suffix in succ_layer_name: handler_key = 'Linear' # Input to FC2
                    elif "conv_layers" in succ_layer_name: handler_key = 'Conv1d' # Input to CNN
                    elif successor_info['type'] == NodeType.EMBEDDING: handler_key = 'Linear'

                    if use_mha_posthook:
                        try:
                            parent_attn_name = succ_layer_name.rsplit('.', 1)[0]  # encoder.layers.X.self_attn
                            mha_module = self.get_module(parent_attn_name)
                            hook_func = make_mha_posthook(self, node_id, successor_info, device)
                            self.hooks.append(mha_module.register_forward_hook(hook_func))
                        except KeyError: print(f"ERROR: MHA module {parent_attn_name} not found for posthook (Node {node_id})")
                        except Exception as e: print(f"ERROR attaching MHA posthook for node {node_id}: {e}")
                    elif handler_key:
                        try:
                            module_to_hook = self.get_module(succ_layer_name)
                            hook_func = make_prehook(self, node_id, successor_info, device, handler_key)
                            self.hooks.append(module_to_hook.register_forward_pre_hook(hook_func))
                        except KeyError: print(f"ERROR: Module {succ_layer_name} not found for standard prehook (Node {node_id})")
                        except Exception as e: print(f"ERROR attaching standard prehook for node {node_id}: {e}")
                    else:
                        print(f"Warning: Unhandled PREFIX node {node_id}. Successor: {succ_layer_name}")

                nodes_processed.add(node_id)


            elif info['type'] == NodeType.POSTFIX:
                # --- Find MODULE Predecessor (using BFS backwards) ---
                predecessor_module_node = None
                predecessor_info = None
                queue = list(self.G.pred[node_id])
                visited = {node_id} | set(queue)
                while queue:
                    pred_node_candidate = queue.pop(0)
                    pred_info_candidate = self.get_node_info(pred_node_candidate)
                    if pred_info_candidate['type'] == NodeType.MODULE:
                        predecessor_module_node = pred_node_candidate
                        predecessor_info = pred_info_candidate
                        break
                    for next_pred in self.G.pred[pred_node_candidate]:
                        if next_pred not in visited:
                            visited.add(next_pred)
                            queue.append(next_pred)
                # --- ---

                if predecessor_info is None:
                    print(f"ERROR: POSTFIX node {node_id} had no preceding MODULE node.")
                    nodes_processed.add(node_id)
                    continue

                pred_layer_name = predecessor_info.get('layer', '')
                # print(f"DEBUG Node {node_id}: Found MODULE predecessor Node {predecessor_module_node}, Layer: {pred_layer_name}")

                # Determine handler key based on predecessor module type
                handler_key = 'Linear' # Default
                try:
                    pred_module = self.get_module(pred_layer_name)
                    if isinstance(pred_module, nn.Conv1d): handler_key = 'Conv1d'
                    elif isinstance(pred_module, nn.Conv2d): handler_key = 'Conv2d'
                    # Add more specific types if needed

                    hook_func = make_posthook(self, node_id, predecessor_info, device, handler_key)
                    self.hooks.append(pred_module.register_forward_hook(hook_func))
                    #print(f"Node {node_id}: Attached standard POST-hook to {pred_layer_name} (handler: {handler_key})")

                except KeyError: print(f"ERROR: Module {pred_layer_name} not found for POSTFIX hook (Node {node_id})")
                except Exception as e: print(f"ERROR attaching POSTFIX hook for node {node_id}: {e}")

                nodes_processed.add(node_id)

        print(f"--- Finished adding/registering hooks. Total PyTorch hooks: {len(self.hooks)}. Registry size: {len(self._hook_registry)} ---")

    
    def clear_hooks(self):
        """ Clear graph hooks. """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.mha_input_buffer = {} # <<< ADD THIS: Clear buffer too
        self.hooked_mha_modules = set() # <<< ADD THIS: Reset hooked set
        self._hook_registry = {} # Clear registry
        print(f"Hooks cleared. Count: {len(self.hooks)}")
    
    def _register_hook_callback(self, module_instance, hook_key, callback):
        """Helper to register a hook in the temporary registry."""
        module_id = id(module_instance)
        if module_id not in self._hook_registry:
            self._hook_registry[module_id] = {}
        self._hook_registry[module_id][hook_key] = callback
        
    
    # def compute_intermediates(self, x, attn_mask=None):
    #     """ Computes all intermediates in a graph network. Takes in a torch tensor (e.g., a batch). """
    #     self.model = self.model.eval()
    #     #print(f"input speech shape is {x[0].shape}")
    #     # this uses the hooks added in add_hooks()
    #     with torch.no_grad(), torch.cuda.amp.autocast():
    #         self.intermediates = {}
    #         if attn_mask != None:
    #             self.model(x, attention_mask=attn_mask)
    #         else:
    #             self.model(x)
    #         sorted_intermediates = sorted(self.intermediates.items(), key=lambda item: item[0])
    #         print("Intermediates captured:")
    #         print(f"DEBUG: Forward pass complete. Captured intermediates for nodes: {sorted(list(self.intermediates.keys()))}")

    #         # Check for missing nodes
    #         prefix_nodes = {n for n, info in self.G.nodes(data=True) if info['type'] == NodeType.PREFIX}
    #         captured_nodes = set(self.intermediates.keys())
    #         missing_prefix = prefix_nodes - captured_nodes
    #         if missing_prefix:
    #             print(f"DEBUG: WARNING - Missing intermediates for PREFIX nodes: {sorted(list(missing_prefix))}")


    #         # for node, tensor in self.intermediates.items():
    #         #     print(f"Node {node}, Shape: {tensor.shape}")
    #         return self.intermediates
    def compute_intermediates(self, x, attn_mask=None, device="cuda"):
        """ Computes all intermediates using attached hooks and registered callbacks. """
        if not isinstance(x, list): # If input is single tensor, wrap in list
            x = [x]
        # Ensure input tensors are on the correct device
        x = [t.to(device) for t in x]
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        self.model = self.model.eval().to(device) # Ensure model is on correct device and in eval mode

        # Add/register hooks *before* the forward pass
        self.add_hooks(device=device)
        self.intermediates = {} # Reset before forward pass


        # --- Pass the hook registry to the relevant layers ---
        # Requires TransformerSentenceEncoderLayer to have _graph_hook_registry_ref attribute
        for module in self.model.modules():
             # Check for the specific type defined in your wav2vec2_model.py
             # You might need to import it here or use isinstance checks carefully
             # Example check (replace with actual import/check):
             is_tsel = type(module).__name__ == 'TransformerSentenceEncoderLayer'
             if is_tsel:
                  # print(f"DEBUG: Setting registry ref for {module}")
                  module._graph_hook_registry_ref = self._hook_registry
             # Ensure MHA modules have their storage ready (it's cleared in their forward)
             is_mha = type(module).__name__ == 'MultiheadAttention'
             if is_mha and not hasattr(module, '_hook_storage'):
                 module._hook_storage = {} # Initialize if missing (should be done in __init__)

    


        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            #print(f"Starting model forward pass for intermediate computation...")
            try:
                # Use the first element if x was originally a single tensor
                input_data = x[0] if len(x) == 1 else x
                if attn_mask is not None:
                    _ = self.model(input_data, attention_mask=attn_mask)
                else:
                    _ = self.model(input_data)
                #print(f"Model forward pass completed.")
            except Exception as e:
                print(f"ERROR during model forward pass in compute_intermediates: {e}")
                import traceback
                traceback.print_exc()
                # Potentially clear hooks even on error
                self.clear_hooks()
                # Clean up registry ref
                for module in self.model.modules():
                     if hasattr(module, '_graph_hook_registry_ref'):
                          module._graph_hook_registry_ref = None
                raise # Re-raise the exception


        # --- Clean up the registry reference ---
        for module in self.model.modules():
             if hasattr(module, '_graph_hook_registry_ref'):
                 module._graph_hook_registry_ref = None

        #print(f"--- compute_intermediates finished. Captured intermediates for {len(self.intermediates)} nodes ---")
        #print(f"DEBUG: Captured Nodes: {sorted(list(self.intermediates.keys()))}")

        # --- Verification ---
        all_hookable_nodes = {n for n, info in self.G.nodes(data=True) if info['type'] in [NodeType.PREFIX, NodeType.POSTFIX]}
        captured_nodes = set(self.intermediates.keys())
        missing_nodes = all_hookable_nodes - captured_nodes
        if missing_nodes:
            print(f"WARNING: Missing intermediates for hookable nodes: {sorted(list(missing_nodes))}")
            # for missing_node in sorted(list(missing_nodes)):
            #      print(f"  - Node {missing_node}: {self.get_node_info(missing_node)}")

        # Optional: Clear hooks immediately after computation if not needed further
        # self.clear_hooks()

        return self.intermediates

    
    @abstractmethod
    def graphify(self):
        """ 
        Abstract method. This function is implemented by your architecture graph file, and is what actually
        creates the graph for your model. 
        """
        return NotImplemented
    
