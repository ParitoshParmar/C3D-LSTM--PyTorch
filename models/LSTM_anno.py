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

import torch
import torch.nn as nn
from opts import *

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed)

class LSTM_anno(nn.Module):
    def __init__(self):
        super(LSTM_anno, self).__init__()
        # defining encoder LSTM layers
        self.rnn = nn.LSTM(4096, 256, 1, batch_first=True)
        self.fc_final_score = nn.Linear(256,1)

    def forward(self, x):
        state = None
        lstm_output, state = self.rnn(x, state)
        final_score = self.fc_final_score(lstm_output[:,-1,:])
        return final_score