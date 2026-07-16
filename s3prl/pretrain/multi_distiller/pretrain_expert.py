"""
    Pre-train expert for distiller
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""

from easydict import EasyDict as edict
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pretrain.multi_distiller.dataset import OnlineWaveDataset
from upstream.multi_distiller.model import MultiDistillerConfig, MultiDistillerModel
from transformers import AutoModel, AutoConfig
import torchaudio
import torchaudio.transforms as transforms
from transformers import ASTForAudioClassification
from transformers import AutoProcessor, ASTModel, ASTConfig
from pretrain.multi_distiller.convert_ssast_dict import convert_ssast_state_dict_to_astmodel
from pretrain.multi_distiller.disable_dropout import disable_MERT_encoder_dropout, disable_AST_encoder_dropout, disable_SSAST_encoder_dropout

import pdb
import collections




class TemporalAligner(nn.Module):
    def __init__(self, max_length_in_seconds=10, input_sample_rate=16000, distilhubert_frame_shift=20, ssast_frame_shift=10):
        """
        TemporalAligner for aligning the time dimension of SSAST and distilHuBERT.
        
        Args:
            max_length_in_seconds: Maximum length for SSAST (in seconds).
            input_sample_rate: The sample rate of the input audio (default 16 kHz).
            distilhubert_frame_shift: The frame shift (in ms) for distilHuBERT features.
            ssast_frame_shift: The frame shift (in ms) for SSAST features.
        """
        super(TemporalAligner, self).__init__()

        # Compute the number of samples for SSAST's max input length
        self.max_length_in_samples = max_length_in_seconds * input_sample_rate
        
        # Frame shifts in samples for SSAST and distilHuBERT
        self.distilhubert_frame_shift_samples = int((distilhubert_frame_shift / 1000) * input_sample_rate)
        self.ssast_frame_shift_samples = int((ssast_frame_shift / 1000) * input_sample_rate)
        
        # Average pooling for temporal downsampling (matching distilHuBERT with SSAST)
        self.temporal_pooling = nn.AvgPool1d(kernel_size=2, stride=2)
    
    def forward(self, ssast_features, distilhubert_features):
        """
        Align the SSAST and distilHuBERT features.
        
        Args:
            ssast_features: The feature tensor from SSAST (batch, time, feature_dim).
            distilhubert_features: The feature tensor from distilHuBERT (batch, time, feature_dim).
            
        Returns:
            Aligned distilHuBERT features cropped and temporally downsampled.
        """
        # Step 1: Perform temporal downsampling of SSAST features

        ssast_features_pooled = self.temporal_pooling(ssast_features.transpose(1, 2)).transpose(1, 2)
        
        # Step 2: Crop distilHuBERT features if they exceed the SSAST max length
        # Determine the maximum number of frames SSAST can process (10 seconds)
        # This may not be needed anymore as I am usnig the interpolation method. Will this be good or bad for pre-training?
        max_frames_ssast = ssast_features_pooled.shape[1]
        max_frames_distilhubert = distilhubert_features.shape[1]
        
        # Crop distilHuBERT features to match the SSAST max frames
        if max_frames_distilhubert > max_frames_ssast:
            distilhubert_features_cropped = distilhubert_features[:, :max_frames_ssast, :]
        else:
            distilhubert_features_cropped = distilhubert_features
        
        if max_frames_distilhubert < max_frames_ssast:
            ssast_features_pooled = ssast_features_pooled[:, :max_frames_distilhubert, :]
        
        
        return ssast_features_pooled, distilhubert_features_cropped





# from audiossl.models.atst.atst import ATST
# from ......MERT.mert_fairseq.models.mert.mert_model import MERTConfig

def freeze_model(model):
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False

def remap_keys(state_dict, prefix):
    """Remap keys in the state_dict to match the model's expected structure."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.backbone'):
            new_key = key.replace('module.backbone', 'encoder')
        elif key.startswith('module.head'):
            new_key = key.replace('module.head', 'projector')  # Adjust based on your model's needs
        elif key.startswith('module.'):
            new_key = key.replace('module.', '', 1)
        else:
            new_key = key
        new_state_dict[f'{prefix}.{new_key}'] = value
    return new_state_dict

def get_ATST_teacher_model(arch, ncrops, atst_model_path, target_device):
    """Configure the ATST model by loading weights, disabling dropouts, and freezing the model."""
    # Initialize the model
    kwargs = {}  # Additional arguments if needed
    teacher_3 = ATST(arch=arch, ncrops=ncrops, **kwargs)

    # Load the full state dictionary from the file
    full_state_dict = torch.load(atst_model_path, map_location='cpu')

    # Extract the 'student' and 'teacher' state dictionaries
    student_state_dict = full_state_dict.get('student', {})
    teacher_state_dict = full_state_dict.get('teacher', {})

    # Remap keys
    student_state_dict = remap_keys(student_state_dict, 'student')
    teacher_state_dict = remap_keys(teacher_state_dict, 'teacher')

    # Combine the remapped state dictionaries
    combined_state_dict = {**student_state_dict, **teacher_state_dict}

    # Load the combined state dict into your model with strict=False to allow minor mismatches
    try:
        missing_keys, unexpected_keys = teacher_3.load_state_dict(combined_state_dict, strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")

    # Move the model to the desired GPU
    teacher_3.to(target_device)

    # Disable dropouts in the student encoder's blocks
    for block in teacher_3.student.encoder.blocks:
        block.attn.attn_drop.p = 0.0
        block.attn.proj_drop.p = 0.0
        block.mlp.drop.p = 0.0
        if hasattr(block, 'drop_path') and isinstance(block.drop_path, torch.nn.Dropout):
            block.drop_path.p = 0.0

    # Disable dropouts in the teacher encoder's blocks
    for block in teacher_3.teacher.encoder.blocks:
        block.attn.attn_drop.p = 0.0
        block.attn.proj_drop.p = 0.0
        block.mlp.drop.p = 0.0
        if hasattr(block, 'drop_path') and isinstance(block.drop_path, torch.nn.Dropout):
            block.drop_path.p = 0.0

    print("[ATST] - Disabled all dropouts in the encoder's blocks")

    # Freeze the parameters of the teacher and student models

    return teacher_3

def convert_mert_state_dict(state_dict):
    new_state_dict_mert = {}
    for key, value in state_dict.items():
        # Convert "conv_layers.0.conv.weight" to "conv_layers.0.0.weight"
        # Convert "conv_layers.0.layer_norm.weight" to "conv_layers.0.2.weight" (assuming layer_norm is at index 2)
        if "conv_layers" in key:
            # Handle the convolution layers
            if "conv.weight" in key:
                new_key = key.replace("conv.weight", "0.weight")
            # Handle the normalization layers
            elif "layer_norm" in key:
                new_key = key.replace("layer_norm.weight", "2.weight").replace("layer_norm.bias", "2.bias")
            # Handle activation layers if needed (you can add this if distilHuBERT expects it)
            else:
                new_key = key
            new_state_dict_mert[new_key] = value
        else:
            new_state_dict_mert[key] = value
    return new_state_dict_mert

def rename_attention_keys_ast(state_dict):
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key

        # Map "ASTSelfAttention" keys to unified format (similar to "self_attn" in other models)
        new_key = new_key.replace("attention.attention.query", "self_attn.q_proj")
        new_key = new_key.replace("attention.attention.key", "self_attn.k_proj")
        new_key = new_key.replace("attention.attention.value", "self_attn.v_proj")
        new_key = new_key.replace("attention.output.dense", "self_attn.out_proj")

        # Map "ASTSelfOutput" layer norm to self-attention layer norm
        new_key = new_key.replace("attention.output", "self_attn_layer_norm")

        # Map "ASTIntermediate" to fc1 and "ASTOutput" to fc2 (for feed-forward network)
        new_key = new_key.replace("intermediate.dense", "fc1")
        new_key = new_key.replace("output.dense", "fc2")

        # Handle the final layer normalization renaming
        new_key = new_key.replace("layernorm_before", "self_attn_layer_norm")
        new_key = new_key.replace("layernorm_after", "final_layer_norm")

        new_state_dict[new_key] = state_dict[key]

    return new_state_dict

def average_weights(mapped_state_dicts):
    """Averages the weights from multiple state_dicts."""
    avg_dict = collections.OrderedDict()

    keys = mapped_state_dicts[0].keys()
    for key in keys:
        weights = [sd[key] for sd in mapped_state_dicts]
        avg_dict[key] = torch.mean(torch.stack(weights), dim=0)

    return avg_dict




class UpstreamPretrainExpert(nn.Module):
    """
    The Distiller pretrain expert
    """

    def __init__(
        self, datarc, upstream_config, device="cuda", multi_gpu=False, **kwargs
    ):
        super().__init__()

        self.datarc = datarc
        self.device = device
        self.multi_gpu = multi_gpu

        if type(upstream_config) == str:
            self.upstream_config = yaml.load(
                open(upstream_config, "r"), Loader=yaml.FullLoader
            )
            print(
                "[UpstreamPretrainExpert] - Using upstream config from:",
                upstream_config,
            )
        elif type(upstream_config) == dict:
            self.upstream_config = upstream_config
            print(
                "[UpstreamPretrainExpert] - Using upstream config from the previous experiment."
            )
        else:
            raise ValueError

        self._get_train_dataloader()

        print("[UpstreamPretrainExpert] - Initializing model...")
        model_config = MultiDistillerConfig(self.upstream_config["multi_distiller"],**datarc)
        self.model = MultiDistillerForPretrain(
            model_config, edict(self.upstream_config["teacher"]) ### here we get the multidistiller part and the teacher part of the file
        )

        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            print(
                "[UpstreamPretrainExpert] - Multi-GPU training Enabled: "
                + str(torch.cuda.device_count())
            )
        print(
            "[UpstreamPretrainExpert] - Number of parameters: "
            + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        )

    def _get_train_dataloader(self):
        dataset = OnlineWaveDataset(
            self.upstream_config["task"],
            self.datarc["train_batch_size"],
            target_level=self.upstream_config["audio"]["target_level"],
            **self.datarc,
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=1,  # for bucketing
            shuffle=True,
            num_workers=self.datarc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
    
    def _get_dev_dataloader(self):
        dataset = OnlineWaveDataset(
            self.upstream_config["task"],
            self.datarc["dev_batch_size"],
            target_level=self.upstream_config["audio"]["target_level"],
            data_type='dev',
            sets=self.datarc["devsets"],
            **self.datarc,
        )
        #sampler = DistributedSampler(dataset) if is_initialized() else None

        self.devloader = DataLoader(
            dataset,
            batch_size=1,  # for bucketing
            shuffle=False,
            num_workers=self.datarc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn
        )

    # Interface
    def load_model(self, all_states):
        if self.multi_gpu:
            self.model.module.distiller.load_state_dict(all_states["Distiller"])
        else:
            self.model.distiller.load_state_dict(all_states["Distiller"])

    # Interface
    def add_state_to_save(self, all_states):
        all_states["Distiller"] = (
            self.model.float().distiller.state_dict()
            if not self.multi_gpu
            else self.model.float().module.distiller.state_dict()
        )
        all_states["Config"] = self.upstream_config
        return all_states

    # Interface
    def get_train_dataloader(self):
        return self.dataloader

    # Interface
    def forward(self, data, records={}, global_step=0, log_step=1000, **kwargs):
        """
        Args:
            data:
                [wave_input, pad_mask]

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

        Return:
            loss
        """

        wave_input, wave_orig, wave_len, pad_mask = data
        wave_input = wave_input.to(self.device)
        wave_len = wave_len.to(self.device)
        pad_mask = pad_mask.type(wave_input.dtype).to(self.device)

        loss, other_res = self.model(
            wave_input,
            wave_orig,
            wave_len,
            pad_mask,
            return_other=global_step % log_step == 0,
        )

        if global_step % log_step == 0:
            for key, value in other_res.items():
                if isinstance(value, torch.Tensor):
                    value = float(value.mean().cpu().item())
                records[key] = value

        return loss, records

    # interface
    def on_before_zero_grad(self):
        pass

    # interface
    def log_records(self, records, logger, prefix, global_step, **kwargs):
        """
        Args:
            records:
                defaultdict(list), contents already appended

            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            prefix:
                used to indicate downstream and train/test on Tensorboard
                eg. 'phone/train-'

            global_step:
                global_step in runner, which is helpful for Tensorboard logging
        """
        for key, values in records.items():
            if isinstance(values, torch.Tensor) and len(values.shape) > 1:
                logger.add_image(f"{prefix}{key}", values, global_step=global_step)
            elif isinstance(values, float):
                logger.add_scalar(f"{prefix}{key}", values, global_step=global_step)


class MultiDistillerForPretrain(nn.Module):
    """
    Distiller for pretraining with flexible number of teacher models.
    """

    def __init__(self, config: MultiDistillerConfig, teacher_config: edict):
        super().__init__()
        self.config = config
        self.distiller = MultiDistillerModel(config)
        #print(f"the distiller model arch inside MultiDistillerForPretrain is {self.distiller}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher_config = teacher_config
        print(f"the teacher config inside MultiDistillerForPretrain is {self.teacher_config}")
        self.teachers = teacher_config.models  # Expecting a list of teacher model names
        

        # Dictionary to store teacher models and processors
        self.teacher_models = {}
        self.teacher_processors = {}

        # Load teacher models based on self.teachers
        for model_name in self.teachers:
            if model_name == 'hubert_base':
                teacher_1 = torch.hub.load("s3prl/s3prl",model_name).to(device)
                if model_name.find("hubert") >= 0 or model_name.find("wav2vec2") >= 0:
                    teacher_1.model.encoder.layerdrop = 0
                    print("[HuBERT] - Disabled teacher's encoder layerdrop")
                    self.temporal_alignment = None
                self.teacher_models[model_name] = teacher_1
            elif model_name == 'mert_v0_public':
                self.temporal_alignment = None
                temp_config = AutoConfig.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)
                temp_config.output_hidden_states = True  # Enable hidden states in the output
                if not hasattr(temp_config, 'conv_pos_batch_norm'):
                    temp_config.conv_pos_batch_norm = False
                teacher_2 = AutoModel.from_pretrained("m-a-p/MERT-v0-public", config=temp_config, trust_remote_code=True).to(device)
                disable_MERT_encoder_dropout(teacher_2)
                self.teacher_models[model_name] = teacher_2
            elif model_name == 'ast':
                temp_config = AutoConfig.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
                temp_config.output_hidden_states = True  # Enable output of hidden states
                #self.temporal_alignment = TemporalAligner()
                #temp_config.max_length = 1598
                temp_config.ignore_mismatched_sizes = True
                teacher_3 = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", config=temp_config).to(device)
                teacher_3_processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
                print("teacher_3_processor needs to be updated!")
                teacher_3_processor.do_normalize = True # I need to modify this later as well.
                teacher_3_processor.mean = self.config.fbank_mean
                teacher_3_processor.std = self.config.fbank_std
                print(f"teacher_3_processor is {teacher_3_processor}")

                print(f"ONCE THIS IS DONE I CAN THEN DISTILL.....")
                teacher_3_processor.return_attention_mask = True
                #teacher_3_processor.max_length = 1598
                #teacher_3_processor.ignore_mismatched_sizes = True
                ####  ########
                print(f"WE NEED TO UNDERSTAND WELL THIS AUTOPROCESSOR!!!.")
                disable_AST_encoder_dropout(teacher_3)
                self.teacher_models[model_name] = teacher_3
                self.teacher_processors[model_name] = teacher_3_processor
            
            elif model_name == "ssast-patch":
                ast_config = ASTConfig(
                    architectures=["ASTModel"],
                    frequency_stride=16,
                    time_stride=16,
                    hidden_size=768,
                    max_length=1024,
                    num_attention_heads=12,
                    num_hidden_layers=12,
                    num_mel_bins=128,
                    qkv_bias=True,
                    output_hidden_states = True
                    )
                teacher_3 = ASTModel(config=ast_config)
                teacher_3_processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
                ssast_state_dict = torch.load("./ssast_checkpoints/SSAST-Base-Patch-400.pth")
                self.temporal_alignment = TemporalAligner()
                teacher_3_processor.do_normalize = True # I need to modify this later as well.
                teacher_3_processor.mean = self.config.fbank_mean
                teacher_3_processor.std = self.config.fbank_std
                print(f"teacher_3_processor is {teacher_3_processor}")
                converted = convert_ssast_state_dict_to_astmodel(ssast_state_dict)
                teacher_3.load_state_dict(converted, strict=True)
                teacher_3 = teacher_3.to("cuda")
                disable_AST_encoder_dropout(teacher_3)
                self.teacher_models[model_name] = teacher_3
                self.teacher_processors[model_name] = teacher_3_processor

            elif model_name == "ssast-frame-local":
                #teacher_3 = ASTModel(config=ast_config)
                from pretrain.multi_distiller.ast_models import ASTModel as ASTModelLocal
                teacher_3 = ASTModelLocal(fstride=128,
                    fshape=128, 
                    tshape=2,
                    tstride=2,
                    input_tdim=1024,
                    input_fdim=128,
                    pretrain_stage=False,
                    model_size='base',
                    load_pretrained_mdl_path="./ssast_checkpoints/SSAST-Base-Frame-400.pth")
                
                teacher_3_processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
                self.temporal_alignment = TemporalAligner()
                teacher_3_processor.do_normalize = True # I need to modify this later as well.
                teacher_3_processor.mean = self.config.fbank_mean
                teacher_3_processor.std = self.config.fbank_std
                teacher_3 = teacher_3.to("cuda")
                disable_SSAST_encoder_dropout(teacher_3)
                self.teacher_models[model_name] = teacher_3
                self.teacher_processors[model_name] = teacher_3_processor
            
            elif model_name == "ssast-frame":
                ast_config = ASTConfig(
                    frequency_stride=128,
                    fshape=128, 
                    tshape=2,
                    time_stride=2,
                    patch_size=16,
                    hidden_size=768,
                    max_length=1024,
                    num_attention_heads=12,
                    num_hidden_layers=12,
                    num_mel_bins=128,
                    qkv_bias=True,
                    output_hidden_states = True
                    )
                teacher_3 = ASTModel(config=ast_config)
                teacher_3_processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
                ssast_state_dict = torch.load("./ssast_checkpoints/SSAST-Base-Frame-400.pth")
                self.temporal_alignment = TemporalAligner()
                teacher_3_processor.do_normalize = True # I need to modify this later as well.
                #teacher_3_processor.mean = self.config.fbank_mean
                #teacher_3_processor.std = self.config.fbank_std
                print(f"teacher_3_processor is {teacher_3_processor}")
                converted = convert_ssast_state_dict_to_astmodel(ssast_state_dict)
                #converted['embeddings.position_embeddings'] = converted['embeddings.position_embeddings'][:, :507, :]
                teacher_3.load_state_dict(converted, strict=True)
                teacher_3 = teacher_3.to("cuda")
                disable_AST_encoder_dropout(teacher_3)
                self.teacher_models[model_name] = teacher_3
                self.teacher_processors[model_name] = teacher_3_processor


            else:
                print(f"Warning: Unknown teacher model {model_name} specified.")
            
        
        # Freeze all teacher models
        for teacher in self.teacher_models.values():
            freeze_model(teacher)
        
        # Initialize loss function
        if config.loss_type == "l1":
            self.loss_func = nn.L1Loss(reduction="none")
        elif config.loss_type == "l2":
            self.loss_func = nn.MSELoss(reduction="none")
        elif config.loss_type == "barlow":
            from pretrain.multi_distiller.loss_functions import BarlowTwinsLoss
            self.loss_func = BarlowTwinsLoss(
                lambda_coeff=config.lambda_coeff,
                lambda_coeff_self_corr=config.lambda_coeff_self_corr,
                number_of_predictions=config.n_tasks,
                scale=config.barlow_scale,
                off_diag_cross_cor_scale=config.frame_level_barlow_scale,
                diag_scale=config.diagonal_scale,
                self_correlation=config.self_correlation,
                use_snr_info=False,
                off_diag_self_cor_scale=config.off_diag_self_cor_scale
            )
            print("[DistillerForPretrain] - Using Barlow Twins loss with cross and self correlation")
        elif config.loss_type == "barlow_old":
            from pretrain.multi_distiller.loss_functions import BarlowTwinsLoss_old
            self.loss_func = BarlowTwinsLoss_old(
                lambda_coeff=config.lambda_coeff,
                lambda_coeff_self_corr=config.lambda_coeff_self_corr,
                number_of_predictions=config.n_tasks,
                scale=config.barlow_scale,
                off_diag_cross_cor_scale=config.frame_level_barlow_scale,
                diag_scale=config.diagonal_scale,
                self_correlation=config.self_correlation,
                use_snr_info=False,
                off_diag_self_cor_scale=config.off_diag_self_cor_scale
            )
            print("[DistillerForPretrain] - Using OLD Barlow Twins loss (default self-corr scale=50000, no fp32 cast)")
        else:
            raise NotImplementedError(config.loss_type)

        self.cosine_loss = config.cosine_loss
        if self.cosine_loss > 0:
            print("[DistillerForPretrain] - Enabled cosine similarity loss.")
        

        # Load pre-saved initial model state if specified (for reproducible task arithmetic)
        if config.init_model_state:
            print(f"[MultiDistillerForPretrain] - Loading initial model state from {config.init_model_state}")
            init_state = torch.load(config.init_model_state, map_location='cpu')
            # Try strict load first; if key mismatch (e.g. single-teacher base loaded
            # into multi-teacher model), fall back to strict=False so the backbone
            # (conv, post_extract_proj, encoder) inits from theta_base while the
            # output heads keep their random init.
            try:
                self.distiller.load_state_dict(init_state)
                print(f"[MultiDistillerForPretrain] - Loaded pre-saved base model (strict, skipping teacher weight init)")
            except RuntimeError as e:
                print(f"[MultiDistillerForPretrain] - Strict load failed: {e}")
                print(f"[MultiDistillerForPretrain] - Retrying with strict=False (backbone from base, output heads random init)")
                missing, unexpected = self.distiller.load_state_dict(init_state, strict=False)
                if missing:
                    print(f"[MultiDistillerForPretrain] - Missing keys (randomly initialized): {missing}")
                if unexpected:
                    print(f"[MultiDistillerForPretrain] - Unexpected keys (ignored from base): {unexpected}")
        else:
            # Ensure that we can only load weights from hubert_base or mert_v0_public
            model_to_initialize = self.config.initialize_from[0]
            if model_to_initialize == 'ast':
                raise AssertionError("[Error] Cannot initialize weights from 'ast' model. The student's architecture is compatible only with 'hubert_base' or 'mert_v0_public'.")
            elif model_to_initialize == 'hubert_base':
                print(f"Initializing student model from {model_to_initialize}...")
                self.load_teacher_weights('hubert_base')
            elif model_to_initialize == 'mert_v0_public':
                print(f"Initializing student model from {model_to_initialize}...")
                self.load_teacher_weights('mert_v0_public')
            elif model_to_initialize == 'all':
                self.load_teacher_weights('all')



    def load_teacher_weights(self, teacher_name, device="cuda"):
        """
        Load the weights from a specified teacher model (hubert_base or mert_v0_public).
        """
        print(f"teacher_name {teacher_name}")
        if teacher_name == "all":
            teacher_model_1 = self.teacher_models.get("mert_v0_public")
            teacher_model_2 = self.teacher_models.get("hubert_base")
            teacher_model_3 = self.teacher_models.get("ssast-frame")

            if teacher_model_1 is None:
                temp_config = AutoConfig.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)
                temp_config.output_hidden_states = True  # Enable hidden states in the output
                if not hasattr(temp_config, 'conv_pos_batch_norm'):
                    temp_config.conv_pos_batch_norm = False
                teacher_model_1 = AutoModel.from_pretrained("m-a-p/MERT-v0-public", config=temp_config, trust_remote_code=True).to(device)
                disable_MERT_encoder_dropout(teacher_model_1)
            if teacher_model_2 is None:
                teacher_model_2 = torch.hub.load("s3prl/s3prl","hubert_base").to(device)
                teacher_model_2.model.encoder.layerdrop = 0
                print("[HuBERT] - Disabled teacher's encoder layerdrop")
            if teacher_model_3 is None:
                ast_config = ASTConfig(
                    frequency_stride=128,
                    fshape=128, 
                    tshape=2,
                    time_stride=2,
                    patch_size=16,
                    hidden_size=768,
                    max_length=1024,
                    num_attention_heads=12,
                    num_hidden_layers=12,
                    num_mel_bins=128,
                    qkv_bias=True,
                    output_hidden_states = True
                    )
                teacher_model_3 = ASTModel(config=ast_config)
                teacher_3_processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
                ssast_state_dict = torch.load("./ssast_checkpoints/SSAST-Base-Frame-400.pth")
                self.temporal_alignment = TemporalAligner()
                teacher_3_processor.do_normalize = True # I need to modify this later as well.
                teacher_3_processor.mean = self.config.fbank_mean
                teacher_3_processor.std = self.config.fbank_std
                print(f"teacher_3_processor is {teacher_3_processor}")
                converted = convert_ssast_state_dict_to_astmodel(ssast_state_dict)
                #converted['embeddings.position_embeddings'] = converted['embeddings.position_embeddings'][:, :507, :]
                #pdb.set_trace()
                teacher_model_3.load_state_dict(converted, strict=True)
                teacher_model_3 = teacher_model_3.to("cuda")
                disable_AST_encoder_dropout(teacher_model_3)

            # Get the state_dict of each model
            mert_state_dict = teacher_model_1.feature_extractor.state_dict()
            mert_state_dict = convert_mert_state_dict(mert_state_dict)

            hubert_state_dict = teacher_model_2.model.feature_extractor.state_dict()


            # Average all weights for the 'all' case
            averaged_conv_layers = average_weights([hubert_state_dict, mert_state_dict])
            

            # Load the averaged state_dict into the student model
            # Handle the conv layers specifically with just MERT and Hubert
            if self.config.init_teacher_conv_layers:
                print(f"[DistillerForPretrain] - Averaging conv layers from Hubert and MERT")
                self.distiller.feature_extractor.load_state_dict(averaged_conv_layers)

                if self.distiller.post_extract_proj is not None:
                    averaged_proj = average_weights([teacher_model_2.model.post_extract_proj.state_dict(), teacher_model_1.feature_projection.projection.state_dict()])
                    if self.distiller.post_extract_proj.weight.shape == averaged_proj['weight'].shape:
                        self.distiller.post_extract_proj.load_state_dict(averaged_proj)
                    else:
                        print(f"[DistillerForPretrain] - Skipping post_extract_proj init (shape mismatch)")
            
            # Load weights for encoder layers
            if self.config.init_teacher_encoder_layers:
                # MERT has `conv`, `padding`, `activation`, distilHuBERT has indices `0`, `1`, `2` in a Sequential
                mert_pos_conv = teacher_model_1.encoder.pos_conv_embed.state_dict()

                # Create a new state_dict for distilHuBERT by mapping the keys
                # Create a new state_dict to map MERT's keys to distilHuBERT's keys
                pos_conv_dict = {
                    '0.bias': mert_pos_conv['conv.bias'],            # Mapping MERT's conv.bias to 0.bias
                    '0.weight_g': mert_pos_conv['conv.weight_g'],    # Mapping MERT's weight_g to 0.weight_g
                    '0.weight_v': mert_pos_conv['conv.weight_v']     # Mapping MERT's weight_v to 0.weight_v
                }
        

                print(f"[DistillerForPretrain] - Loading encoder positional convolution from MERT")
                # hubert style: teacher_model_2.model.encoder.pos_conv.state_dict()
                average_pos_conv = average_weights([teacher_model_2.model.encoder.pos_conv.state_dict(),pos_conv_dict])
                self.distiller.encoder.pos_conv.load_state_dict(average_pos_conv)

                #### mising to load the transformer part with the weights of the 3 transormer teacher models.
                ##### here I need to mix the three of them!.
                print(f"[DistillerForPretrain] - Loading encoder layers from MERT, hubert and ssast")
                for l in range(self.config.encoder_layers):
                    # Mapping MERT's HubertEncoderLayer to distilHuBERT's TransformerSentenceEncoderLayer
                    mert_encoder_layer = teacher_model_1.encoder.layers[l].state_dict()
                    # Create a new state dict with mapped keys for distilHuBERT
                    new_encoder_layer_dict = {}
                    for key, value in mert_encoder_layer.items():
                        # Rename attention block
                        if 'attention.' in key:
                            new_key = key.replace('attention.', 'self_attn.')
                        # Rename layer_norm to self_attn_layer_norm
                        elif 'layer_norm' in key and 'final_layer_norm' not in key:
                            new_key = key.replace('layer_norm', 'self_attn_layer_norm')
                        elif 'final_layer_norm' in key:
                            new_key = key  # No changes for final_layer_norm
                        # Rename feed forward layers
                        elif 'feed_forward.intermediate_dense' in key:
                            new_key = key.replace('feed_forward.intermediate_dense', 'fc1')
                        elif 'feed_forward.output_dense' in key:
                            new_key = key.replace('feed_forward.output_dense', 'fc2')
                        else:
                            new_key = key  # If no changes are needed, keep the key the same
                        # Add the mapped key and value to the new dict
                        new_encoder_layer_dict[new_key] = value
                    
                        #### hubert one : teacher_model.model.encoder.layers[l].state_dict()
                        ### ssast one:  converted_state_dict_ast = rename_attention_keys_ast(teacher_model_3.encoder.layer[l].state_dict())
                    
                    converted_state_dict_ast = rename_attention_keys_ast(teacher_model_3.encoder.layer[l].state_dict())
                    hubert_encoder_layer = teacher_model_2.model.encoder.layers[l].state_dict()

                    averaged_encoder = average_weights([new_encoder_layer_dict, converted_state_dict_ast,hubert_encoder_layer])
                    self.distiller.encoder.layers[l].load_state_dict(averaged_encoder)

        else:
            teacher_model = self.teacher_models.get(teacher_name)
            if teacher_model is None:
                print(f"teacher_name is {teacher_name} and self.config.initialize_from is {self.config.initialize_from[0]} ")
                if teacher_name == self.config.initialize_from[0]:
                    if teacher_name == "mert_v0_public":
                        temp_config = AutoConfig.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)
                        temp_config.output_hidden_states = True  # Enable hidden states in the output
                        if not hasattr(temp_config, 'conv_pos_batch_norm'):
                            temp_config.conv_pos_batch_norm = False
                        teacher_model = AutoModel.from_pretrained("m-a-p/MERT-v0-public", config=temp_config, trust_remote_code=True).to(device)
                        disable_MERT_encoder_dropout(teacher_model)
                    if teacher_name == "hubert_base":
                        teacher_model = torch.hub.load("s3prl/s3prl","hubert_base").to(device)
                        teacher_model.model.encoder.layerdrop = 0
                        print("[HuBERT] - Disabled teacher's encoder layerdrop")
                

                else:
                    raise ValueError(f"[Error] Teacher model '{teacher_name}' not found in the loaded teacher models.")

        # Example: loading weights from hubert_base or mert_v0_public for feature extractor
        if teacher_name == 'hubert_base':
            print(f"[DistillerForPretrain] - Loading weights from {teacher_name}")
            
            # Load weights for feature extractor
            if self.config.init_teacher_conv_layers:
                print(f"[DistillerForPretrain] - Initializing feature extractor from {teacher_name}")
                self.distiller.feature_extractor.load_state_dict(
                    teacher_model.model.feature_extractor.state_dict()
                )
                if self.distiller.post_extract_proj is not None:
                    teacher_proj = teacher_model.model.post_extract_proj.state_dict()
                    if self.distiller.post_extract_proj.weight.shape == teacher_proj['weight'].shape:
                        self.distiller.post_extract_proj.load_state_dict(teacher_proj)
                    else:
                        print(f"[DistillerForPretrain] - Skipping post_extract_proj init (shape mismatch: "
                              f"student={list(self.distiller.post_extract_proj.weight.shape)}, "
                              f"teacher={list(teacher_proj['weight'].shape)})")

            # Load weights for encoder layers
            if self.config.init_teacher_encoder_layers:
                print(f"[DistillerForPretrain] - Initializing encoder from {teacher_name}")
                self.distiller.encoder.pos_conv.load_state_dict(
                    teacher_model.model.encoder.pos_conv.state_dict()
                )
                for l in range(self.config.encoder_layers):
                    self.distiller.encoder.layers[l].load_state_dict(
                        teacher_model.model.encoder.layers[l].state_dict()
                    )

        if teacher_name == 'mert_v0_public':
            print(f"[DistillerForPretrain] - Loading weights from {teacher_name}")
            # Load weights for feature extractor
            # Retrieve the state_dict of the MERT feature extractor
            state_dict = teacher_model.feature_extractor.state_dict()

            # Modify the keys to match distilHuBERT's expected layer names
            new_state_dict_mert = convert_mert_state_dict(state_dict)

            if not teacher_name == "all":
                if self.config.init_teacher_conv_layers:
                    print(f"[DistillerForPretrain] - Initializing feature extractor from {teacher_name}")
                    self.distiller.feature_extractor.load_state_dict(new_state_dict_mert)

                    if self.distiller.post_extract_proj is not None:
                        teacher_proj = teacher_model.feature_projection.projection.state_dict()
                        if self.distiller.post_extract_proj.weight.shape == teacher_proj['weight'].shape:
                            self.distiller.post_extract_proj.load_state_dict(teacher_proj)
                        else:
                            print(f"[DistillerForPretrain] - Skipping post_extract_proj init (shape mismatch: "
                                  f"student={list(self.distiller.post_extract_proj.weight.shape)}, "
                                  f"teacher={list(teacher_proj['weight'].shape)})")
            

            # Load weights for encoder layers
            if self.config.init_teacher_encoder_layers:
                # MERT has `conv`, `padding`, `activation`, distilHuBERT has indices `0`, `1`, `2` in a Sequential
                mert_pos_conv = teacher_model.encoder.pos_conv_embed.state_dict()

                # Create a new state_dict for distilHuBERT by mapping the keys
                # Create a new state_dict to map MERT's keys to distilHuBERT's keys
                pos_conv_dict = {
                    '0.bias': mert_pos_conv['conv.bias'],            # Mapping MERT's conv.bias to 0.bias
                    '0.weight_g': mert_pos_conv['conv.weight_g'],    # Mapping MERT's weight_g to 0.weight_g
                    '0.weight_v': mert_pos_conv['conv.weight_v']     # Mapping MERT's weight_v to 0.weight_v
                }
        

                print(f"[DistillerForPretrain] - Loading encoder positional convolution from MERT")
                self.distiller.encoder.pos_conv.load_state_dict(pos_conv_dict)
                
                print(f"[DistillerForPretrain] - Loading encoder layers from MERT")
                for l in range(self.config.encoder_layers):
                    # Mapping MERT's HubertEncoderLayer to distilHuBERT's TransformerSentenceEncoderLayer
                    mert_encoder_layer = teacher_model.encoder.layers[l].state_dict()
                    # Create a new state dict with mapped keys for distilHuBERT
                    new_encoder_layer_dict = {}
                    for key, value in mert_encoder_layer.items():
                        # Rename attention block
                        if 'attention.' in key:
                            new_key = key.replace('attention.', 'self_attn.')
                        # Rename layer_norm to self_attn_layer_norm
                        elif 'layer_norm' in key and 'final_layer_norm' not in key:
                            new_key = key.replace('layer_norm', 'self_attn_layer_norm')
                        elif 'final_layer_norm' in key:
                            new_key = key  # No changes for final_layer_norm
                        # Rename feed forward layers
                        elif 'feed_forward.intermediate_dense' in key:
                            new_key = key.replace('feed_forward.intermediate_dense', 'fc1')
                        elif 'feed_forward.output_dense' in key:
                            new_key = key.replace('feed_forward.output_dense', 'fc2')
                        else:
                            new_key = key  # If no changes are needed, keep the key the same
                        # Add the mapped key and value to the new dict
                        new_encoder_layer_dict[new_key] = value



                    self.distiller.encoder.layers[l].load_state_dict(new_encoder_layer_dict)



    def forward(
        self,
        wave_input: torch.Tensor,
        wave_orig: list,
        wave_len: torch.Tensor,
        pad_mask: torch.Tensor,
        return_other: bool = False,
    ):
        """
        Forward function.
        """
        feat, feat_final, pred, pad_mask = self.distiller(wave_input, pad_mask)

        teachers_hidden_states = {}
        with torch.no_grad():
            wave_orig = [wave.to(wave_input.device) for wave in wave_orig]
            if isinstance(wave_orig, list):
                    max_length = max(wave.size(0) for wave in wave_orig)
                    padded_wave_orig = [F.pad(wave, (0, max_length - wave.size(0))) for wave in wave_orig]
                    wave_orig = torch.stack(padded_wave_orig).to(wave_input.device)
            with torch.cuda.amp.autocast(False):
                # Loop through the teacher models to gather hidden states
                for model_name, teacher in self.teacher_models.items():
                    if model_name == 'hubert_base':
                        teacher_hiddens = teacher(wave_orig)
                    elif model_name == 'mert_v0_public':
                        teacher_hiddens = teacher(wave_orig)
                    elif model_name == 'ast' or model_name.startswith("ssast"):
                        inputs = self.teacher_processors[model_name](wave_orig.cpu().numpy(), sampling_rate=16000, return_tensors="pt",return_attention_mask=True)
                        if model_name =="ssast-frame-local":
                            teacher_hiddens, _ = teacher(inputs["input_values"].cuda())
                            teacher_hiddens = torch.stack(teacher_hiddens)
                            padded_hidden_states = F.pad(teacher_hiddens, (0, 0, 0, 0, 0, 0, 1, 0)) # Adds one dimension from 12 to 13 at the start
                            teacher_hiddens = {"hidden_states": padded_hidden_states}
                        else:
                            inputs = {key: value.to('cuda:0') for key, value in inputs.items()}
                            teacher_hiddens = teacher(**inputs)

                    # Extract hidden states based on task embedding type
                    if self.config.task_emb_type in ["expand-last", "hnet", "self-hidden"]:
                        teacher_hiddens = [
                            teacher_hiddens["hidden_states"][i]
                            for i in self.distiller.pred_layer_id
                        ]
                        teachers_hidden_states[model_name] = torch.stack(teacher_hiddens, dim=1)


        # Apply average pooling if enabled (for Barlow Twins loss)
        if self.config.use_average_pooling and hasattr(self.distiller, 'avg_pooling') and self.distiller.avg_pooling is not None:
            # pred shape: B x N x T x D
            pred = self.distiller.avg_pooling(pred)
            for teacher_key in teachers_hidden_states:
                teachers_hidden_states[teacher_key] = self.distiller.avg_pooling(teachers_hidden_states[teacher_key])

        # Compute all objectives
        (
            total_loss,
            rec_loss,
            rec_layer_loss_dict,
            feat_pen,
            sim_loss,
            sim_layer_loss_dict,
        ) = self.compute_loss(feat, pred, teachers_hidden_states, return_other)

        if return_other:
            with torch.no_grad():
                other_res = {
                    "rec_loss": rec_loss,
                    "feat_pen": feat_pen,
                    "sim_loss": sim_loss,
                    "norm_feat_final": feat_final.pow(2).mean(),
                }

                # Initialize a dictionary to keep norms for each teacher
                teacher_norms = {}

                # Calculate norms for each teacher and add to teacher_norms
                for model_name, hidden_states in teachers_hidden_states.items():
                    teacher_norms[model_name] = torch.abs(hidden_states).mean((0, 2, 3))

                # Log metrics for each teacher
                for model_name, norm in teacher_norms.items():
                    # Retrieve the layer-wise losses from the dictionaries
                    rec_layer_loss = rec_layer_loss_dict.get(model_name, None)
                    sim_layer_loss = sim_layer_loss_dict.get(model_name, None)

                    if rec_layer_loss is not None:
                        # If task_emb_type is 'none', log only the first layer as before
                        if self.config.task_emb_type == "none":
                            other_res[f"rec_l_{model_name}_{self.config.n_tasks}"] = rec_layer_loss[0]
                            other_res[f"tar_norm_l_{model_name}_{self.config.n_tasks}"] = norm[0]
                            if sim_layer_loss is not None:
                                other_res[f"sim_l_{model_name}_{self.config.n_tasks}"] = sim_layer_loss[0]
                        else:
                            # Otherwise, log all layers or based on pred_layer_id
                            for i in range(min(self.config.n_tasks, len(rec_layer_loss))):
                                layer_id = i + 1
                                if self.config.task_emb_type in ["expand-last", "hnet", "self-hidden"]:
                                    layer_id = self.distiller.pred_layer_id[i]

                                # Logging for each layer of each teacher
                                other_res[f"rec_l_{model_name}_{layer_id}"] = rec_layer_loss[i]
                                other_res[f"tar_norm_l_{model_name}_{layer_id}"] = norm[i]
                                if sim_layer_loss is not None and i < len(sim_layer_loss):
                                    other_res[f"sim_l_{model_name}_{layer_id}"] = sim_layer_loss[i]

                # Additional task embedding logging if applicable
                if self.config.task_emb_type not in ["expand-last", "hnet", "self-hidden"]:
                    other_res["norm_task_emb"] = self.distiller.task_embedding.weight.pow(2).mean()
        else:
            other_res = None



        return total_loss, other_res

    def compute_loss(self, feat, pred, target, return_other=False):
        """
        Computes loss for multiple teachers.
        Inputs:
            feat: B x T x D
            pred: Dict containing predictions from multiple teachers
            target: Dict containing targets corresponding to each teacher
            return_other: Flag to indicate if additional losses should be returned
        """
        # Initialize variables to accumulate losses
        total_loss = 0
        total_rec_loss = 0
        total_sim_loss = 0
        total_feat_pen = 0

        rec_layer_loss_dict = {}
        sim_layer_loss_dict = {}

        #print(f"fix here for when you use more teachers....")
        # Iterate over each teacher's predictions and targets
        for teacher_key in target.keys(): ## on the meantime.... this needs to be fixed
            if type(pred) == dict:
                teacher_pred = pred[teacher_key]
            else:
                teacher_pred = pred    # [teacher_key]  # Prediction from the current teacher
            teacher_target = target[teacher_key]  # Target corresponding to the current teacher
            
            aligned_preds = []  # To store aligned student features
            aligned_targets = []  # To store aligned teacher features

            if self.temporal_alignment:
                for i in range(pred.shape[1]): ### do this outside... is better and more efficient, capitalize one of the for already being done outside...
                    align_teacher, align_student = self.temporal_alignment(teacher_target[:,i,:,:], teacher_pred[:,i,:,:])
                    # Append the aligned features to the lists
                    aligned_preds.append(align_student.unsqueeze(1))  # Add back the layer dimension
                    aligned_targets.append(align_teacher.unsqueeze(1))  # Add back the layer dimension

                # Concatenate aligned layers back to 4D tensors (batch, layers, time, feature_dim)
                teacher_pred = torch.cat(aligned_preds, dim=1)
                teacher_target = torch.cat(aligned_targets, dim=1)
                    




            # Ensure shapes match
            assert teacher_pred.shape == teacher_target.shape, (teacher_pred.shape, teacher_target.shape)

            # Compute reconstruction loss
            # Check if using Barlow Twins loss (returns N-dimensional loss)
            from pretrain.multi_distiller.loss_functions import BarlowTwinsLoss, BarlowTwinsLoss_old
            if isinstance(self.loss_func, (BarlowTwinsLoss, BarlowTwinsLoss_old)):
                # Barlow loss expects B x N x T x D and returns N x T loss
                if self.config.loss_style == "frame":
                    rec_loss = self.loss_func(teacher_pred, teacher_target)  # N x T
                else:
                    # Utterance level - mean over time
                    rec_loss = self.loss_func(
                        torch.mean(teacher_pred, dim=2),
                        torch.mean(teacher_target, dim=2)
                    )  # N
                total_rec_loss += rec_loss.mean()

                # Optionally compute layer-wise reconstruction loss
                if return_other:
                    with torch.no_grad():
                        if self.config.loss_style == "frame":
                            rec_layer_loss = rec_loss.mean(1)  # Per-layer loss
                        else:
                            rec_layer_loss = rec_loss
                    rec_layer_loss_dict[teacher_key] = rec_layer_loss
                else:
                    rec_layer_loss_dict[teacher_key] = None
            else:
                # Standard L1/L2 loss
                rec_loss = self.loss_func(teacher_pred, teacher_target)  # B x N x T x D
                total_rec_loss += rec_loss.mean()

                # Optionally compute layer-wise reconstruction loss
                if return_other:
                    with torch.no_grad():
                        rec_layer_loss = rec_loss.mean((0, 2, 3))  # Per-layer reconstruction loss
                    rec_layer_loss_dict[teacher_key] = rec_layer_loss
                else:
                    rec_layer_loss_dict[teacher_key] = None

            # Compute cosine similarity loss if applicable
            if self.cosine_loss > 0:
                sim_loss = -F.logsigmoid(F.cosine_similarity(teacher_pred, teacher_target, dim=-1))  # B x N x T
                total_sim_loss += sim_loss.mean()

                # Optionally compute layer-wise similarity loss
                if return_other:
                    with torch.no_grad():
                        sim_layer_loss = sim_loss.mean((0, 2))  # Per-layer similarity loss
                    sim_layer_loss_dict[teacher_key] = sim_layer_loss
                else:
                    sim_layer_loss_dict[teacher_key] = None
            else:
                sim_layer_loss = 0
                sim_layer_loss_dict[teacher_key] = None

            # Compute feature penalty loss
            feat_pen = feat.float().pow(2).mean()
            total_feat_pen += feat_pen

        # Sum up the total loss components
        total_loss = (
            total_rec_loss
            + total_feat_pen * self.config.feat_pen_loss
            + total_sim_loss * self.cosine_loss
        )

        # print("=====================================", total_loss)
        return total_loss, total_rec_loss, rec_layer_loss_dict, total_feat_pen, total_sim_loss, sim_layer_loss_dict