runner:
  n_epochs: 1
  total_steps: -1 # changed from 200000
  gradient_clipping: 5.0
  gradient_accumulate_steps: 1 ### make this bigger to have 24 size
  num_workers: 24
  log_step: 50
  save_step: 10000
  max_keep: 10

  fp16: true

optimizer:
  name: AdamW_with_schedule
  lr: 2.e-4
  warmup_proportion: 0.07
  betas: [0.9, 0.98]
  eps: 1.e-6
  weight_decay: 1.e-6

pretrain_expert:
  datarc:
    num_workers: 24
    train_batch_size: 24 #### the train batch size is too small... this will make it ultra noisy
    dev_batch_size: 24
    max_timestep: 0
    libri_root: /path/to/LibriSpeech
    file_path: ./data/len_for_bucket
    sets: ['train-clean-100', 'train-clean-360', 'train-other-500']
    devsets: ['dev-clean']
    data_stats:    # Add new fields for mean and std
      fbank_mean: [-8.202537669959721]
      fbank_std: [4.238643955336016]
      wav_mean:  [8.7623e-13]
      wav_std:  [0.0608]



### Librispeech 100 stats:    
#   fbank_mean: #[-7.840440776167798]     # These will be filled with the computed values later
    #   fbank_std: #[4.131988934508892]
    #   wav_mean: #[-1.1761e-12]
    #   wav_std: #[0.06037661]

### Librispeech 960 stats:    
# fbank_mean: -8.202537669959721
# fbank_std: 4.238643955336016
# wav_mean:  8.7623e-13
# wav_std:  0.0608

#####music4all stats:
#Final dataset mean_value_fbank: -5.10452733130284, Final dataset std_value_fbank: 4.961768594778897
#Final dataset wav_mean: 7.513496769195882e-12, Final dataset wav_std: 0.2172156125307083
# fbank_mean: -5.10452733130284
# fbank_std: 4.961768594778897
# wav_mean:  7.513496769195882e-12
# wav_std:  0.2172156125307083