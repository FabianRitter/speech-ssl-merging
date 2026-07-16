import torch

def convert_ssast_state_dict_to_astmodel(pretrained_dict, layers: int = 12):
    conversion_dict = {
        'module.v.cls_token': 'embeddings.cls_token',
        'module.v.dist_token': 'embeddings.distillation_token',
        'module.v.pos_embed': 'embeddings.position_embeddings',
        'module.v.patch_embed.proj.weight': 'embeddings.patch_embeddings.projection.weight',
        'module.v.patch_embed.proj.bias': 'embeddings.patch_embeddings.projection.bias',
        'module.v.norm.weight': 'layernorm.weight',
        'module.v.norm.bias': 'layernorm.bias',
    }

    for i in range(layers):
        conversion_dict[
            f'module.v.blocks.{i}.norm1.weight'] = f'encoder.layer.{i}.layernorm_before.weight'
        conversion_dict[
            f'module.v.blocks.{i}.norm1.bias'] = f'encoder.layer.{i}.layernorm_before.bias'
        conversion_dict[f'module.v.blocks.{i}.attn.qkv.weight'] = [
            f'encoder.layer.{i}.attention.attention.query.weight',
            f'encoder.layer.{i}.attention.attention.key.weight',
            f'encoder.layer.{i}.attention.attention.value.weight'
        ]
        conversion_dict[f'module.v.blocks.{i}.attn.qkv.bias'] = [
            f'encoder.layer.{i}.attention.attention.query.bias',
            f'encoder.layer.{i}.attention.attention.key.bias',
            f'encoder.layer.{i}.attention.attention.value.bias'
        ]
        conversion_dict[
            f'module.v.blocks.{i}.attn.proj.weight'] = f'encoder.layer.{i}.attention.output.dense.weight'
        conversion_dict[
            f'module.v.blocks.{i}.attn.proj.bias'] = f'encoder.layer.{i}.attention.output.dense.bias'
        conversion_dict[
            f'module.v.blocks.{i}.norm2.weight'] = f'encoder.layer.{i}.layernorm_after.weight'
        conversion_dict[
            f'module.v.blocks.{i}.norm2.bias'] = f'encoder.layer.{i}.layernorm_after.bias'
        conversion_dict[
            f'module.v.blocks.{i}.mlp.fc1.weight'] = f'encoder.layer.{i}.intermediate.dense.weight'
        conversion_dict[
            f'module.v.blocks.{i}.mlp.fc1.bias'] = f'encoder.layer.{i}.intermediate.dense.bias'
        conversion_dict[
            f'module.v.blocks.{i}.mlp.fc2.weight'] = f'encoder.layer.{i}.output.dense.weight'
        conversion_dict[
            f'module.v.blocks.{i}.mlp.fc2.bias'] = f'encoder.layer.{i}.output.dense.bias'

    converted_dict = {}
    for key, value in pretrained_dict.items():
        if key in conversion_dict:
            mapped_key = conversion_dict[key]
            if isinstance(mapped_key, list):
                # Assuming value is split equally among q, k, v if it's a concatenated tensor
                split_size = value.shape[0] // 3
                converted_dict[mapped_key[0]] = value[:split_size]
                converted_dict[mapped_key[1]] = value[split_size:2 * split_size]
                converted_dict[mapped_key[2]] = value[2 * split_size:]
            else:
                converted_dict[mapped_key] = value

    return converted_dict