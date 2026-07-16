import torch
from transformers import AutoModel, AutoConfig
from pretrain.multi_distiller.disable_dropout import disable_MERT_encoder_dropout
import pdb


def freeze_model(model):
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False

def load_hubert_model():
    teacher_1 = torch.hub.load("s3prl/s3prl", "hubert_base").cuda()
    teacher_1.model.encoder.layerdrop = 0
    #freeze_model(teacher_1)
    return teacher_1

def load_mert_model():
    temp_config = AutoConfig.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)
    temp_config.output_hidden_states = True  # Enable hidden states in the output
    teacher_2 = AutoModel.from_pretrained("m-a-p/MERT-v0-public", config=temp_config, trust_remote_code=True).cuda()
    disable_MERT_encoder_dropout(teacher_2)
    #freeze_model(teacher_2)
    return teacher_2

def convert_mert_conv_state_dict(state_dict):
    new_state_dict_mert = {}
    for key, value in state_dict.items():
        if "conv_layers" in key:
            parts = key.split('.')
            # parts = ['conv_layers', layer_idx, submodule, ...]
            layer_idx = parts[1]
            submodule = parts[2]
            rest = parts[3:]
            if submodule == "conv":
                new_key = f"conv_layers.{layer_idx}.0.{'.'.join(rest)}"
            elif submodule == "layer_norm":
                new_key = f"conv_layers.{layer_idx}.2.{'.'.join(rest)}"
            else:
                continue  # Skip activation layers if not needed
            new_state_dict_mert[new_key] = value
        else:
            # Handle any other keys if necessary
            pass
    return new_state_dict_mert

def convert_mert_attn_state_dict(state_dict):
    new_encoder_state_dict = {}
    for key, value in state_dict.items():
        parts = key.split('.')
        # Handle 'pos_conv_embed.conv' keys
        if parts[0] == 'pos_conv_embed' and parts[1] == 'conv':
            # Map to 'pos_conv.0'
            param_name = '.'.join(parts[2:])
            new_key = f'pos_conv.0.{param_name}'
            new_encoder_state_dict[new_key] = value
        # Handle 'layer_norm' at the encoder level
        elif parts[0] == 'layer_norm':
            param_name = '.'.join(parts[1:])
            new_key = f'layer_norm.{param_name}'
            new_encoder_state_dict[new_key] = value
        # Handle transformer layers
        elif parts[0] == 'layers':
            layer_idx = parts[1]
            submodule = parts[2]
            rest = parts[3:]
            if submodule == 'attention':
                attn_submodule = rest[0]
                attn_rest = rest[1:]
                new_key = f"layers.{layer_idx}.self_attn.{attn_submodule}.{'.'.join(attn_rest)}"
                new_encoder_state_dict[new_key] = value
            elif submodule == 'layer_norm':
                # This is the self-attention layer norm
                param_name = '.'.join(rest)
                new_key = f'layers.{layer_idx}.self_attn_layer_norm.{param_name}'
                new_encoder_state_dict[new_key] = value
            elif submodule == 'final_layer_norm':
                # This is the final layer norm
                param_name = '.'.join(rest)
                new_key = f'layers.{layer_idx}.final_layer_norm.{param_name}'
                new_encoder_state_dict[new_key] = value
            elif submodule == 'feed_forward':
                ff_submodule = rest[0]
                ff_rest = rest[1:]
                if ff_submodule == 'intermediate_dense':
                    param_name = '.'.join(ff_rest)
                    new_key = f'layers.{layer_idx}.fc1.{param_name}'
                    new_encoder_state_dict[new_key] = value
                elif ff_submodule == 'output_dense':
                    param_name = '.'.join(ff_rest)
                    new_key = f'layers.{layer_idx}.fc2.{param_name}'
                    new_encoder_state_dict[new_key] = value
                else:
                    print(f"Unrecognized feed_forward submodule: {ff_submodule}")
            else:
                print(f"Unrecognized submodule in layers: {submodule}")
        else:
            print(f"Unrecognized key: {key}")
    return new_encoder_state_dict

def assemble_new_checkpoint(averaged_state_dict):
    new_checkpoint = {}
    new_checkpoint['model_cfg'] = {'_name': 'hubert', 'label_rate': 50.0, 'extractor_mode': 'default', 'encoder_layers': 12, 'encoder_embed_dim': 768, 'encoder_ffn_embed_dim': 3072, 'encoder_attention_heads': 12, 'activation_fn': 'gelu', 'layer_type': 'transformer', 'dropout': 0.1, 'attention_dropout': 0.1, 'activation_dropout': 0.0, 'encoder_layerdrop': 0.05, 'dropout_input': 0.1, 'dropout_features': 0.1, 'final_dim': 256, 'untie_final_proj': False, 'layer_norm_first': False, 'conv_feature_layers': '[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2', 'conv_bias': False, 'logit_temp': 0.1, 'target_glu': False, 'feature_grad_mult': 0.1, 'mask_length': 10, 'mask_prob': 0.8, 'mask_selection': 'static', 'mask_other': 0.0, 'no_mask_overlap': False, 'mask_min_space': 1, 'mask_channel_length': 10, 'mask_channel_prob': 0.0, 'mask_channel_selection': 'static', 'mask_channel_other': 0.0, 'no_mask_channel_overlap': False, 'mask_channel_min_space': 1, 'conv_pos': 128, 'conv_pos_groups': 16, 'latent_temp': [2.0, 0.5, 0.999995], 'skip_masked': False, 'skip_nomask': False, 'checkpoint_activations': False, 'required_seq_len_multiple': 2, 'depthwise_conv_kernel_size': 31, 'attn_type': '', 'pos_enc_type': 'abs', 'fp16': True}
    new_checkpoint["dictionaries_symbols"] = [['<s>', '<pad>', '</s>', '<unk>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238', '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252', '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266', '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280', '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294', '295', '296', '297', '298', '299', '300', '301', '302', '303', '304', '305', '306', '307', '308', '309', '310', '311', '312', '313', '314', '315', '316', '317', '318', '319', '320', '321', '322', '323', '324', '325', '326', '327', '328', '329', '330', '331', '332', '333', '334', '335', '336', '337', '338', '339', '340', '341', '342', '343', '344', '345', '346', '347', '348', '349', '350', '351', '352', '353', '354', '355', '356', '357', '358', '359', '360', '361', '362', '363', '364', '365', '366', '367', '368', '369', '370', '371', '372', '373', '374', '375', '376', '377', '378', '379', '380', '381', '382', '383', '384', '385', '386', '387', '388', '389', '390', '391', '392', '393', '394', '395', '396', '397', '398', '399', '400', '401', '402', '403', '404', '405', '406', '407', '408', '409', '410', '411', '412', '413', '414', '415', '416', '417', '418', '419', '420', '421', '422', '423', '424', '425', '426', '427', '428', '429', '430', '431', '432', '433', '434', '435', '436', '437', '438', '439', '440', '441', '442', '443', '444', '445', '446', '447', '448', '449', '450', '451', '452', '453', '454', '455', '456', '457', '458', '459', '460', '461', '462', '463', '464', '465', '466', '467', '468', '469', '470', '471', '472', '473', '474', '475', '476', '477', '478', '479', '480', '481', '482', '483', '484', '485', '486', '487', '488', '489', '490', '491', '492', '493', '494', '495', '496', '497', '498', '499']]
    new_checkpoint['model_weight'] = averaged_state_dict  # Set to None or omit if not needed
    new_checkpoint["task_cfg"] = {'_name': 'hubert_pretraining', 'data': '/checkpoint/wnhsu/data/librispeech/960h/iter/250K_50hz_km100_mp0_65_v2', 'fine_tuning': False, 'labels': ['layer6.km500'], 'label_dir': None, 'label_rate': 50.0, 'sample_rate': 16000, 'normalize': False, 'enable_padding': False, 'max_keep_size': None, 'max_sample_size': 250000, 'min_sample_size': 32000, 'single_target': False, 'random_crop': True, 'pad_audio': False}
    return new_checkpoint


def average_weights(state_dict_list):
    """Averages the weights from multiple state_dicts."""
    avg_state_dict = {}
    # Find common keys
    keys = set(state_dict_list[0].keys())
    for sd in state_dict_list[1:]:
        keys &= set(sd.keys())

    for key in keys:
        weights = [sd[key] for sd in state_dict_list]
        avg_state_dict[key] = torch.mean(torch.stack(weights), dim=0)

    return avg_state_dict


def average_models(save_path):
    teacher_1 = load_hubert_model()
    teacher_2 = load_mert_model()

    # **Feature Extractor (Convolutional Layers)**
    hubert_conv_state_dict = teacher_1.model.feature_extractor.state_dict()
    mert_conv_state_dict_raw = teacher_2.feature_extractor.state_dict()
    mert_conv_state_dict = convert_mert_conv_state_dict(mert_conv_state_dict_raw)
    averaged_conv_layers = average_weights([hubert_conv_state_dict, mert_conv_state_dict])

    # **Feature Projection**
    hubert_proj_state_dict = teacher_1.model.post_extract_proj.state_dict()
    mert_proj_state_dict = teacher_2.feature_projection.projection.state_dict()
    averaged_proj = average_weights([hubert_proj_state_dict, mert_proj_state_dict])

    # **Encoder Layers**
    hubert_encoder_state_dict = teacher_1.model.encoder.state_dict()
    mert_encoder_state_dict_raw = teacher_2.encoder.state_dict()
    mert_encoder_state_dict = convert_mert_attn_state_dict(mert_encoder_state_dict_raw)
    averaged_encoder_layers = average_weights([hubert_encoder_state_dict, mert_encoder_state_dict])


    # **Assemble the New State Dict**
    averaged_state_dict = {}
    # Add averaged convolutional layers
    for key, value in averaged_conv_layers.items():
        averaged_state_dict[f'feature_extractor.{key}'] = value

    # Add averaged feature projection
    for key, value in averaged_proj.items():
        averaged_state_dict[f'post_extract_proj.{key}'] = value

    # Add averaged encoder layers
    for key, value in averaged_encoder_layers.items():
        averaged_state_dict[f'encoder.{key}'] = value

    # Add remaining parameters from HuBERT (e.g., 'layer_norm', 'mask_emb')
    hubert_full_state_dict = teacher_1.model.state_dict()
    for key, value in hubert_full_state_dict.items():
        if key not in averaged_state_dict:
            averaged_state_dict[key] = value

    
    new_checkpoint = assemble_new_checkpoint(averaged_state_dict)

    # **Load Averaged State Dict into HuBERT Model**
    #from s3prl.upstream.hubert.expert import UpstreamExpert as HuBERTUpstreamExpert
    #averaged_model = HuBERTUpstreamExpert(ckpt=None, model_config=None)
    #averaged_model.model.load_state_dict(averaged_state_dict, strict=False)

    # **Save the Averaged Model**
    torch.save(new_checkpoint, save_path)
    print(f"Averaged model saved to {save_path}")
    
    
if __name__ == "__main__":
    save_path = 'result/pretrain/hubert-mert-teachers-average/averaged_model.ckpt'
    average_models(save_path)

