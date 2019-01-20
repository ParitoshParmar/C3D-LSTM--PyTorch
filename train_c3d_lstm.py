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


import os
import torch
from torch.utils.data import DataLoader
from data_loader import VideoDataset
import numpy as np
import random
from models.C3D import C3D
import scipy.stats as stats
import torch.optim as optim
import torch.nn as nn
from models.LSTM_anno import LSTM_anno

from opts import *


torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True

def save_model(model, model_name, epoch, path):
    model_path = os.path.join(path, '%s_%d.pth' % (model_name, epoch))
    torch.save(model.state_dict(), model_path)

def train_phase(train_dataloader, optimizer, criterion, epoch):
    model_CNN.eval()
    model_lstm.train()

    iteration = 0
    for data in train_dataloader:
        with torch.no_grad():
            true_final_score = data['label_final_score'].unsqueeze_(1).type(torch.FloatTensor).cuda()
            video = data['video'].transpose_(1, 2).cuda()

            batch_size, C, frames, H, W = video.shape
            clip_feats = torch.Tensor([]).cuda()

            for i in np.arange(0, frames - 17, 16):
                clip = video[:, :, i:i + 16, :, :]
                clip_feats_temp = model_CNN(clip)
                clip_feats_temp.unsqueeze_(0)
                clip_feats_temp.transpose_(0, 1)
                clip_feats = torch.cat((clip_feats, clip_feats_temp), 1)

        pred_final_score = model_lstm(clip_feats)

        loss = criterion(pred_final_score, true_final_score)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 20 == 0:
            print('Epoch: ', epoch, ' Iter: ', iteration, ' Loss: ', loss, end="")
            print(' ')
        iteration += 1


def test_phase(test_dataloader):
    with torch.no_grad():
        pred_scores = []; true_scores = []

        model_CNN.eval()
        model_lstm.eval()

        for data in test_dataloader:
            true_scores.extend(data['label_final_score'].data.numpy())
            video = data['video'].transpose_(1, 2).cuda()

            batch_size, C, frames, H, W = video.shape
            clip_feats = torch.Tensor([]).cuda()

            for i in np.arange(0, frames - 17, 16):
                clip = video[:, :, i:i + 16, :, :]
                clip_feats_temp = model_CNN(clip)
                clip_feats_temp.unsqueeze_(0)
                clip_feats_temp.transpose_(0, 1)
                clip_feats = torch.cat((clip_feats, clip_feats_temp), 1)

            temp_final_score = model_lstm(clip_feats)
            pred_scores.extend([element[0] for element in temp_final_score.data.cpu().numpy()])

        rho, p = stats.spearmanr(pred_scores, true_scores)
        print('Predicted scores: ', pred_scores)
        print('True scores: ', true_scores)
        print('Correlation: ', rho)


def main():

    parameters_2_optimize = (list(model_lstm.parameters()))

    optimizer = optim.Adam(parameters_2_optimize, lr=0.0001)

    criterion = nn.MSELoss()

    train_dataset = VideoDataset('train')
    test_dataset = VideoDataset('test')
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


    # actual training, testing loops
    for epoch in range(10):
        saving_dir = '...'
        if epoch == 0:  # save models every 5 epochs
            save_model(model_lstm, 'model_my_lstm', epoch, saving_dir)

        print('-------------------------------------------------------------------------------------------------------')

        train_phase(train_dataloader, optimizer, criterion, epoch)
        test_phase(test_dataloader)
        if (epoch+1) % model_ckpt_interval == 0: # save models every 5 epochs
            save_model(model_lstm, 'model_my_lstm', epoch, saving_dir)

        # lr updates
        if (epoch + 1) % global_lr_stepsize == 0:
            learning_rate = learning_rate * global_lr_gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate


if __name__ == '__main__':
    # loading the altered C3D (ie C3D upto before fc-6)
    model_CNN_pretrained_dict = torch.load('c3d.pickle')
    model_CNN = C3D()
    model_CNN_dict = model_CNN.state_dict()
    model_CNN_pretrained_dict = {k: v for k, v in model_CNN_pretrained_dict.items() if k in model_CNN_dict}
    model_CNN_dict.update(model_CNN_pretrained_dict)
    model_CNN.load_state_dict(model_CNN_dict)
    model_CNN = model_CNN.cuda()

    # lstm
    model_lstm = LSTM_anno()
    model_lstm = model_lstm.cuda()

    main()