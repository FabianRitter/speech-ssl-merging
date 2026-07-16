from graphs.base_graph import BIGGraph, NodeType
from graphs.transformer_enc_graph import TransformerEncoderGraph  # Import existing TransformerEncoderGraph


class HuBERTGraph(BIGGraph):
    def __init__(self, model, merge_type='ff+attn', num_layers=None, num_heads=None, model_type="hubert", merge_cnn=False,
                 merge_qkv_input=False,   # Align input space fed INTO Q, K, V
                 merge_attn_output=True, # Align output space AFTER attention computation (before out_proj) - Corresponds to current ff+attn
                 merge_ffn_output=True):   # Align output space AFTER fc1 (before fc2) - Corresponds to current ff+attn)
        super().__init__(model)
        self.merge_type = merge_type
        print(f"merge_type is {self.merge_type}")
        self.model_type = model_type
        self.merge_cnn = merge_cnn
        self.merge_qkv_input = merge_qkv_input
        self.merge_attn_output = merge_attn_output
        self.merge_ffn_output = merge_ffn_output

        # Auto-detect num_layers and num_heads from the model if not provided
        if num_layers is not None:
            self.num_layers = num_layers
        else:
            self.num_layers = self._detect_num_layers(model)

        if num_heads is not None:
            self.num_heads = num_heads
        else:
            self.num_heads = self._detect_num_heads(model)

        print(f"HuBERTGraph: num_layers={self.num_layers}, num_heads={self.num_heads}, model_type={self.model_type}")

        # Set the prefix based on the model type
        # 'hubert' = loaded via s3prl hub (has model. prefix from UpstreamExpert wrapper)
        # 'distilled' = MultiDistillerModel loaded directly (no prefix)
        if model_type == 'hubert':
            prefix = "model."
        else:
            prefix = ""

        # Module mappings for HuBERT-style architectures (shared between teacher and distilled)
        self.modules = {
            'pos_conv': f'{prefix}encoder.pos_conv',
            'q': 'self_attn.q_proj',
            'k': 'self_attn.k_proj',
            'v': 'self_attn.v_proj',
            'lin_attn': 'self_attn.out_proj',
            'attn_ln': 'self_attn_layer_norm',
            'fc1': 'fc1',
            'fc2': 'fc2',
            'final_ln': 'final_layer_norm',
        }

    def _detect_num_layers(self, model):
        """Auto-detect the number of encoder layers by inspecting the model."""
        # Try different paths depending on model wrapping
        for attr_path in ['model.encoder.layers', 'encoder.layers']:
            obj = model
            try:
                for part in attr_path.split('.'):
                    obj = getattr(obj, part)
                n = len(obj)
                print(f"  Auto-detected {n} encoder layers via {attr_path}")
                return n
            except AttributeError:
                continue
        print("  WARNING: Could not auto-detect num_layers, defaulting to 12")
        return 12

    def _detect_num_heads(self, model):
        """Auto-detect the number of attention heads from the first encoder layer."""
        for attr_path in ['model.encoder.layers.0.self_attn.num_heads',
                          'encoder.layers.0.self_attn.num_heads']:
            obj = model
            try:
                for part in attr_path.split('.'):
                    obj = getattr(obj, part)
                print(f"  Auto-detected {obj} attention heads")
                return obj
            except AttributeError:
                continue
        print("  WARNING: Could not auto-detect num_heads, defaulting to 12")
        return 12
    
    def get_downsample_rates(self, key: str) -> int:
        return 320

    def add_conv_block_nodes(self, name_prefix, input_node):
        """Add individual layers of a CNN block to the graph with PREFIX node inserted in sequence."""
        sequential = self.get_module(name_prefix)
        layers_names = [str(idx) for idx in range(len(sequential))] # ['0', '1', '2', '3']

        current_node = input_node

        stable_idx = 2 if "conv_layers.0" in name_prefix else 0

        for idx, layer_name in enumerate(layers_names):
            layer_full_name = f"{name_prefix}.{layer_name}"
            

            module_node = self.create_node(layer_name=layer_full_name, node_type=NodeType.MODULE)
            self.add_directed_edge(current_node, module_node)
            current_node = module_node

            if self.merge_cnn and idx == stable_idx: # Insert PREFIX node after the stable layer
                prefix_layer_name = f"{name_prefix}.{stable_idx}" # Name of the stable layer
                prefix_node = self.create_node(layer_name=prefix_layer_name, node_type=NodeType.PREFIX)
                self.add_directed_edge(current_node, prefix_node) # Connect the stable layer node to the PREFIX node
                current_node = prefix_node # Update current_node to be the PREFIX node


        output_node = current_node # The final node after the loop (either last MODULE or PREFIX if inserted at the end)

        return output_node # Return the final node, which will be PREFIX node if inserted, or last MODULE


    def _get_prefix(self):
        """Return the model prefix based on model_type."""
        return "model." if self.model_type == 'hubert' else ""

    def add_feature_extractor_nodes(self, input_node):
        """
        Adds all feature extractor nodes in sequence (e.g., `conv_layers.0`, `conv_layers.1`).
        """
        prefix = self._get_prefix()
        conv_layers_name = f"{prefix}feature_extractor.conv_layers"

        # Iterate through each block in the conv_layers
        for i, _ in enumerate(self.get_module(conv_layers_name)):
            block_name = f"{conv_layers_name}.{i}"
            input_node = self.add_conv_block_nodes(block_name, input_node)

        return input_node


    

    def add_positional_encoding(self, input_node):
        """Add positional encoding (pos_conv) to the graph."""
        pos_conv_node = self.create_node(
            layer_name=self.modules['pos_conv'],
            node_type=NodeType.MODULE
        )
        self.add_directed_edge(input_node, pos_conv_node)
        return pos_conv_node

    def add_layerblock_nodes(self, name_prefix, input_node, merge_type):
        """
        Add nodes for a single transformer encoder layer in HuBERT.
        """
        modules = self.modules
        residual_attn_input = input_node
        # --- Self-Attention Block ---
        # Create input nodes for Q, K, V potentially with PREFIX nodes
        q_input_node = residual_attn_input
        k_input_node = residual_attn_input
        v_input_node = residual_attn_input

        if merge_type in ['qkv', 'qkv+attn', 'qkv+ff', 'all']:
            # Add PREFIX before Q, K, V projections
            q_prefix_node = self.create_node(node_type=NodeType.PREFIX, layer_name=f"{name_prefix}.{modules['q']}") # Associate layer for context
            self.add_directed_edge(residual_attn_input, q_prefix_node)
            q_input_node = q_prefix_node

            k_prefix_node = self.create_node(node_type=NodeType.PREFIX, layer_name=f"{name_prefix}.{modules['k']}")
            self.add_directed_edge(residual_attn_input, k_prefix_node)
            k_input_node = k_prefix_node

            v_prefix_node = self.create_node(node_type=NodeType.PREFIX, layer_name=f"{name_prefix}.{modules['v']}")
            self.add_directed_edge(residual_attn_input, v_prefix_node)
            v_input_node = v_prefix_node
        
        # Add Q, K, V module nodes
        q_node = self.add_nodes_from_sequence(name_prefix, [modules['q']], q_input_node)
        k_node = self.add_nodes_from_sequence(name_prefix, [modules['k']], k_input_node)
        v_node = self.add_nodes_from_sequence(name_prefix, [modules['v']], v_input_node)


        # Attention computation (represented simplistically by SUM nodes here)
        # Q + K -> Attention Scores -> Attention Weights (implicitly handled by MultiheadAttention)
        attn_qk_node = self.create_node(node_type=NodeType.SUM) # Represents QK interaction point
        self.add_directed_edge(q_node, attn_qk_node)
        self.add_directed_edge(k_node, attn_qk_node)

        # Attention Weights + V -> Weighted Values -> Aggregation (implicitly handled by MultiheadAttention)
        attn_v_node = self.create_node(node_type=NodeType.SUM) # Represents V interaction point
        self.add_directed_edge(attn_qk_node, attn_v_node) # Simplified dependency
        self.add_directed_edge(v_node, attn_v_node)

        # Output Projection (lin_attn / out_proj)
        out_proj_input_node = attn_v_node # Input comes from attention value aggregation

        if merge_type in ['ff+attn', 'qkv+attn', 'all']:
            out_proj_prefix_node = self.create_node(node_type=NodeType.PREFIX, layer_name=f"{name_prefix}.{modules['lin_attn']}")
            self.add_directed_edge(attn_v_node, out_proj_prefix_node)
            out_proj_input_node = out_proj_prefix_node
        
        out_proj_node = self.add_nodes_from_sequence(name_prefix, [modules['lin_attn']], out_proj_input_node)

        # Add residual connection for attention block
        attn_output_node = self.create_node(node_type=NodeType.SUM)
        self.add_directed_edge(out_proj_node, attn_output_node)
        self.add_directed_edge(residual_attn_input, attn_output_node) # Add residual from original input



        # LayerNorm after attention
        ln_attn_node = self.add_nodes_from_sequence(name_prefix, [modules['attn_ln']], attn_output_node)

        # --- Feed-Forward Block ---
        residual_ff_input = ln_attn_node # Store input for residual connection

        # FC1
        fc1_input_node = residual_ff_input
        # Note: PREFIX for fc1 is implicitly handled by the fc2 PREFIX node's merge operation later.
        # If you wanted a separate alignment *after* fc1, you'd add a PREFIX here.
        fc1_node = self.add_nodes_from_sequence(name_prefix, [modules['fc1']], fc1_input_node)

        # FC2 (with potential PREFIX before it)
        fc2_input_node = fc1_node # Input is the output of fc1

        if merge_type in ['ff_only','ff+attn', 'qkv+ff', 'all']: # Align space *before* fc2
            fc2_prefix_node = self.create_node(node_type=NodeType.PREFIX, layer_name=f"{name_prefix}.{modules['fc2']}")
            self.add_directed_edge(fc1_node, fc2_prefix_node)
            fc2_input_node = fc2_prefix_node

        fc2_node = self.add_nodes_from_sequence(name_prefix, [modules['fc2']], fc2_input_node)

        # Add residual connection for feed-forward block
        ff_output_node = self.create_node(node_type=NodeType.SUM)
        self.add_directed_edge(fc2_node, ff_output_node)
        self.add_directed_edge(residual_ff_input, ff_output_node) # Add residual from input to FF block

        # Final Layer Normalization
        final_ln_node = self.add_nodes_from_sequence(name_prefix, [modules['final_ln']], ff_output_node)

        # POSTFIX node (optional, used for residual connections in some merge types in original code)
        # We might not need this explicit POSTFIX if handling residuals directly as above.
        # Let's stick to the PREFIX approach for now.


 
        
        return final_ln_node # Return the final node of the layer block

    
    def add_transformer_encoder(self, input_node):
        """Add the transformer encoder using TransformerEncoderGraph."""
        # Create a TransformerEncoderGraph for HuBERT's transformer layers
        transformer_graph = TransformerEncoderGraph(
            self.model,
            modules=self.modules,
            layer_name='encoder.layers',
            enc_prefix='',
            merge_type=self.merge_type,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            qk=True,
            name='hubert',
            classifier=False
        )

        # Generate the graph starting from the input node
        transformer_graph.reset_graph()
        transformer_graph.G = self.G  # Share the same graph object
        prefix = self._get_prefix()
        layer_prefix = f"{prefix}encoder.layers"

        for layer_idx in range(self.num_layers):
            name_prefix = f"{layer_prefix}.{layer_idx}"
            input_node = self.add_layerblock_nodes(name_prefix, input_node, self.merge_type)
        return input_node
        

    def graphify(self):
        """
        Build the full graph for the HuBERT feature extractor.
        Works for both teacher models (12 layers, via s3prl hub) and
        distilled models (2+ layers, loaded directly).
        """
        # Start with the INPUT node
        input_node = self.create_node(node_type=NodeType.INPUT)

        # Add the feature extractor
        feature_output_node = self.add_feature_extractor_nodes(input_node)

        # Add the post-extraction projection
        prefix = self._get_prefix()
        proj_name = f"{prefix}post_extract_proj"
        proj_node = self.create_node(layer_name=proj_name, node_type=NodeType.MODULE)

        self.add_directed_edge(feature_output_node, proj_node)

        # Step 3: Add positional encoding
        pos_conv_node = self.add_positional_encoding(proj_node)

        # Step 4: Add transformer encoder
        transformer_output = self.add_transformer_encoder(pos_conv_node)

        # Add the OUTPUT node
        output_node = self.create_node(node_type=NodeType.OUTPUT)
        self.add_directed_edge(transformer_output, output_node)

        return self


if __name__ == "__main__":
    import s3prl.hub as hub
    merge_type = 'ff+attn'  # Change this to experiment with other types
    model = getattr(hub, 'hubert_base')()
    device = 'cuda'  # or cpu
    model = model.to(device)
    # Build and visualize the graph
    graph = HuBERTGraph(model, merge_type=merge_type).graphify()
    graph.draw(save_path=f"hubert_graph_{merge_type}.png")

    # Print the node details
    for node in graph.G.nodes:
        print(f"Node {node}: {graph.get_node_info(node)}")
        print(f"Successors: {graph.succs(node)}")
