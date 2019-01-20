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

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
from opts import *
from scipy.io import loadmat

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True

def load_image_train(image_path, hori_flip, transform=None):
    image = Image.open(image_path)
    size = input_resize
    interpolator_idx = random.randint(0,3)
    interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    interpolator = interpolators[interpolator_idx]
    image = image.resize(size, interpolator)
    if hori_flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    size = input_resize
    interpolator_idx = random.randint(0,3)
    interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    interpolator = interpolators[interpolator_idx]
    image = image.resize(size, interpolator)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

class VideoDataset(Dataset):

    def __init__(self, mode):
        super(VideoDataset, self).__init__()
        self.mode = mode
        if self.mode == 'train':
            self.annotations = loadmat('input/consolidated_train_list.mat').get('consolidated_train_list')
        else:
            self.annotations = loadmat('input/consolidated_test_list.mat').get('consolidated_test_list')



    def __getitem__(self, ix):
        action = int(self.annotations[ix][0])
        sample_no = int(self.annotations[ix][1])
        transform = transforms.Compose([transforms.CenterCrop(H),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image_list = sorted((glob.glob(os.path.join(main_datasets_dir, num2action.get(action), 'frames',
                                                    str('{:03d}'.format(sample_no)), '*.jpg'))))
        images = torch.zeros(sample_length, C, H, W)
        hori_flip = 0
        for i in np.arange(0, sample_length):
            if self.mode == 'train':
                hori_flip += random.randint(0,1)
                images[i] = load_image_train(image_list[i], hori_flip, transform)
            else:
                images[i] = load_image(image_list[i], transform)

        label_final_score = self.annotations[ix][2] / 100
        # split_1: train_stats
        if action == 1:
               std = 14.4912
        elif action == 2:
               std = 00.8608
        elif action == 3:
               std = 11.3202
        elif action == 4:
               std = 12.8210
        elif action == 5:
               std = 14.7112
        else:
               std = 15.2136
        label_final_score = label_final_score / std

        data = {}
        data['video'] = images
        data['label_final_score'] = label_final_score
        data['action'] = action

        return data


    def __len__(self):
        print('No. of samples: ', len(self.annotations))
        return len(self.annotations)