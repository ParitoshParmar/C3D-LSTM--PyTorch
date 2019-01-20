# Author: Paritosh Parmar (https://github.com/ParitoshParmar)
#
# Code for C3D-LSTM used in:
# [1] @inproceedings{parmar2017learning,
#   title={Learning to score olympic events},
#   author={Parmar, Paritosh and Morris, Brendan Tran},
#   booktitle={Computer Vision and Pattern Recognition Workshops (CVPRW), 2017 IEEE Conference on},
#   pages={76--84},
#   year={2017},
#   organization={IEEE}}
#
# [2] @article{parmar2018action,
#   title={Action Quality Assessment Across Multiple Actions},
#   author={Parmar, Paritosh and Morris, Brendan Tran},
#   journal={arXiv preprint arXiv:1812.06367},
#   year={2018}}

main_datasets_dir = '...'

diving_dir = 'diving_v1' + '_samples_lstm_103'
gymvault_dir = 'gym_vault' + '_samples_lstm_103'
ski_dir = 'ski_big_air' + '_samples_lstm_103'
snowb_dir = 'snowboard_big_air' + '_samples_lstm_103'
sync3m_dir = 'sync_diving_3m' + '_samples_lstm_103'
sync10m_dir = 'sync_diving_10m' + '_samples_lstm_103'

num2action = {1: diving_dir,
              2: gymvault_dir,
              3: ski_dir,
              4: snowb_dir,
              5: sync3m_dir,
              6: sync10m_dir}

input_resize = 171,128
C,H,W = 3, 112, 112

sample_length = 96

randomseed = 0

train_batch_size = 6
test_batch_size = 5
model_ckpt_interval = 5

global_lr_stepsize = 5
global_lr_gamma = 0.5