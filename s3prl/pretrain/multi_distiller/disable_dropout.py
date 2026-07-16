from easydict import EasyDict as edict
import torch
import torch.nn as nn



def disable_MERT_encoder_dropout(model):
    """Disable all dropouts in the encoder layers of the model by setting their probabilities to 0.0."""

    # Disable encoder layer dropout if available in config
    if hasattr(model.config, 'encoder_layerdrop'):
        model.config.encoder_layerdrop = 0.0  # Set encoder layer dropout to 0
        print("[MERT] - Disabled all dropouts in the encoder's blocks via config")

    # Iterate through all encoder layers and disable their dropouts by setting p=0.0
    for layer in model.encoder.layers:
        # Disable attention dropout
        if hasattr(layer, 'attention') and hasattr(layer.attention, 'dropout'):
            if isinstance(layer.attention.dropout, nn.Dropout):
                layer.attention.dropout.p = 0.0  # Correctly set the probability to 0
            elif isinstance(layer.attention.dropout, float):
                layer.attention.dropout = 0.0  # Directly set the float value

        # Disable intermediate and output dropouts in the feed-forward layer
        if hasattr(layer, 'feed_forward'):
            if hasattr(layer.feed_forward, 'intermediate_dropout'):
                if isinstance(layer.feed_forward.intermediate_dropout, nn.Dropout):
                    layer.feed_forward.intermediate_dropout.p = 0.0
                elif isinstance(layer.feed_forward.intermediate_dropout, float):
                    layer.feed_forward.intermediate_dropout = 0.0  # Directly set the float value

            if hasattr(layer.feed_forward, 'output_dropout'):
                if isinstance(layer.feed_forward.output_dropout, nn.Dropout):
                    layer.feed_forward.output_dropout.p = 0.0
                elif isinstance(layer.feed_forward.output_dropout, float):
                    layer.feed_forward.output_dropout = 0.0  # Directly set the float value

        # Disable general dropout in the layer if applicable
        if hasattr(layer, 'dropout'):
            if isinstance(layer.dropout, nn.Dropout):
                layer.dropout.p = 0.0
            elif isinstance(layer.dropout, float):
                layer.dropout = 0.0  # Directly set the float value

    print("[MERT] - Disabled all dropouts in the encoder's layers by setting p=0.0 where applicable")


def disable_AST_encoder_dropout(model):
    """Disable all dropouts in the encoder layers of the ASTModel by setting their probabilities to 0.0."""

    # Disable encoder layer dropout if available in config
    if hasattr(model.config, 'encoder_layerdrop'):
        model.config.encoder_layerdrop = 0.0  # Set encoder layer dropout to 0

    # Iterate through all encoder layers and disable their dropouts by setting p=0.0
    for layer in model.encoder.layer:  # Access each ASTLayer in the encoder
        # Disable attention dropout within ASTAttention
        if hasattr(layer, 'attention') and hasattr(layer.attention, 'attention'):
            attention = layer.attention.attention
            if hasattr(attention, 'dropout') and isinstance(attention.dropout, nn.Dropout):
                attention.dropout.p = 0.0  # Set attention dropout to 0.0

        # Disable output dropout within ASTSelfOutput
        if hasattr(layer, 'attention') and hasattr(layer.attention, 'output'):
            output = layer.attention.output
            if hasattr(output, 'dropout') and isinstance(output.dropout, nn.Dropout):
                output.dropout.p = 0.0  # Set output dropout to 0.0

        # Disable intermediate dropout within ASTIntermediate
        if hasattr(layer, 'intermediate'):
            intermediate = layer.intermediate
            if hasattr(intermediate, 'intermediate_act_fn'):
                act_fn = intermediate.intermediate_act_fn
                if isinstance(act_fn, nn.Dropout):
                    act_fn.p = 0.0  # Set intermediate activation dropout to 0.0

        # Disable output dropout within ASTOutput
        if hasattr(layer, 'output'):
            output = layer.output
            if hasattr(output, 'dropout') and isinstance(output.dropout, nn.Dropout):
                output.dropout.p = 0.0  # Set output dropout to 0.0

        # Disable any general dropout in the layer if applicable
        if hasattr(layer, 'dropout') and isinstance(layer.dropout, nn.Dropout):
            layer.dropout.p = 0.0

    print("[AST] - Disabled all dropouts in the encoder's layers by setting p=0.0 where applicable")

import torch.nn as nn

def disable_SSAST_encoder_dropout(model):
    """Disable all dropouts in the ASTModel by setting their probabilities to 0.0."""
    # Disable dropout in positional embedding if applicable
    if hasattr(model.v, 'pos_drop') and isinstance(model.v.pos_drop, nn.Dropout):
        model.v.pos_drop.p = 0.0

    # Iterate through all transformer blocks and disable their dropouts
    for i, block in enumerate(model.v.blocks):
        # Disable attention dropout
        if hasattr(block.attn, 'attn_drop') and isinstance(block.attn.attn_drop, nn.Dropout):
            block.attn.attn_drop.p = 0.0

        # Disable projection dropout
        if hasattr(block.attn, 'proj_drop') and isinstance(block.attn.proj_drop, nn.Dropout):
            block.attn.proj_drop.p = 0.0

        # Disable dropout in MLP layers (GELU activation function is not dropout)
        if hasattr(block.mlp, 'drop') and isinstance(block.mlp.drop, nn.Dropout):
            block.mlp.drop.p = 0.0

    print("[AST] - Disabled all applicable dropouts in the model.")