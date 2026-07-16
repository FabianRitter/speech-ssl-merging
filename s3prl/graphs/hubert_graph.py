from graphs.base_graph import BIGGraph, NodeType
from graphs.transformer_enc_graph import TransformerEncoderGraph  # Import existing TransformerEncoderGraph


class HuBERTGraph(BIGGraph):
    def __init__(self, model, merge_type='ff+attn', num_layers=12, num_heads=12, model_type="hubert", merge_cnn=False,
                 merge_qkv_input=False,   # Align input space fed INTO Q, K, V
                 merge_attn_output=True, # Align output space AFTER attention computation (before out_proj) - Corresponds to current ff+attn
                 merge_ffn_output=True):   # Align output space AFTER fc1 (before fc2) - Corresponds to current ff+attn)
        super().__init__(model)
        self.merge_type = merge_type
        print(f"merge_type is {self.merge_type}")
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.model_type = model_type
        self.merge_cnn = merge_cnn
        self.merge_qkv_input = merge_qkv_input
        self.merge_attn_output = merge_attn_output
        self.merge_ffn_output = merge_ffn_output

        # Set the prefix based on the model type
        prefix = "model." if model_type == 'hubert' else ""

        # Module mappings for HuBERT
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


    def add_feature_extractor_nodes(self, input_node):
        """
        Adds all feature extractor nodes in sequence (e.g., `conv_layers.0`, `conv_layers.1`).
        """
        conv_layers_name = "model.feature_extractor.conv_layers" if self.model_type == 'hubert' else "feature_extractor.conv_layers"

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
        residual = input_node

        # Self-Attention Block
        value_node = self.add_nodes_from_sequence(name_prefix, [modules['v']], residual)
        key_node = self.add_nodes_from_sequence(name_prefix, [modules['k']], residual)
        input_node = self.add_nodes_from_sequence(name_prefix, [modules['q'], NodeType.SUM], residual)
        self.add_directed_edge(key_node, input_node)
        input_node = self.add_nodes_from_sequence(name_prefix, [NodeType.SUM], input_node)
        self.add_directed_edge(value_node, input_node)

        # Add PREFIX before lin_attn for ff+attn merge type
        if merge_type in ['ff+attn', 'all']:
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                    [NodeType.PREFIX, modules['lin_attn']], 
                                                    input_node)
        else:
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                    [modules['lin_attn']], 
                                                    input_node)
        # Add SUM for residual connection AFTER lin_attn
        input_node = self.add_nodes_from_sequence(name_prefix, [NodeType.SUM], input_node)
        self.add_directed_edge(residual, input_node)
        
        input_node = self.add_nodes_from_sequence(name_prefix, [modules['attn_ln']], input_node=input_node)
        
        # Residual Connection for Feed-Forward Layers
        if merge_type in ['res_only', 'ff+res', 'all']:
            residual = input_node
        
        # Feed-Forward Block
        input_node = self.add_nodes_from_sequence(name_prefix, [modules['fc1'], NodeType.PREFIX, modules['fc2'], NodeType.SUM], input_node)
        self.add_directed_edge(residual, input_node)

        # Final Layer Normalization
        if merge_type in ['all', 'ff+res', 'res_only']:
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['final_ln'], NodeType.POSTFIX], input_node)
        else:
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['final_ln']], input_node)
        
        return input_node

    
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
        layer_prefix = "model.encoder.layers" if self.model_type == 'hubert' else "encoder.layers"


        for layer_idx in range(self.num_layers):
            name_prefix = f"{layer_prefix}.{layer_idx}"
            input_node = self.add_layerblock_nodes(name_prefix, input_node, self.merge_type)
        return input_node
        

    def graphify(self):
        """
        Build the full graph for the HuBERT feature extractor.
        """
        # Start with the INPUT node
        input_node = self.create_node(node_type=NodeType.INPUT)

        # Add the feature extractor
        feature_output_node = self.add_feature_extractor_nodes(input_node)

        # Add the post-extraction projection
        proj_pefix = "model.post_extract_proj" if self.model_type == 'hubert' else "post_extract_proj"
        proj_node = self.create_node(layer_name=proj_pefix, node_type=NodeType.MODULE)

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
