import argparse
import numpy as np
import cv2
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import FFDNet
import utils
from tqdm import tqdm
import numpy as np
import cv2
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import FFDNet
import utils

def read_image(image_path, is_gray):
    """
    :return: Normalized Image (C * W * H)
    """
    if is_gray:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(image.T, 0) # 1 * W * H
    else:
        image = cv2.imread(image_path)
        image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).transpose(2, 1, 0) # 3 * W * H

    return utils.normalize(image)

def load_images(is_train, is_gray, base_path):
    """
    :param base_path: ./train_data/
    :return: List[Patches] (C * W * H)
    
    if is_gray:
        train_dir = 'gray/train/'
        val_dir = 'gray/val/'
    else:
        train_dir = 'rgb/train/'
        val_dir = 'rgb/val/'
    """
    image_dir = base_path
    print('> Loading images in ' + image_dir)
    images = []
    for fn in next(os.walk(image_dir))[2]:
        image = read_image(image_dir + fn, is_gray)
        images.append(image)
    return images

def images_to_patches(images, patch_size):
    """
    :param images: List[Image (C * W * H)]
    :param patch_size: int
    :return: (n * C * W * H)
    """
    patches_list = []
    for image in images:
        patches = utils.image_to_patches(image, patch_size=patch_size)
        if len(patches) != 0:
            patches_list.append(patches)
    del images
    return np.vstack(patches_list)

def train():
    args = {
        'train_path': '/home/natori21_u/train_data_jaxa/map_train/',
        'is_gray': True,
        'patch_size': 64,
        'train_noise_interval': [0, 75, 15],
        'val_noise_interval': [0, 60, 30],
        'batch_size': 128,
        'epoches': 80,
        'val_epoch': 5,
        'learning_rate': 1e-3,
        'save_checkpoints': 5,
        'model_path': './models/',
        'use_gpu': True
    }
    args['cuda'] = args['use_gpu'] and torch.cuda.is_available()

    train_dataset = load_images(is_train=True, is_gray=args['is_gray'], base_path=args['train_path'])
    val_dataset = load_images(is_train=False, is_gray=args['is_gray'], base_path=args['train_path'])
    print(f'\tTrain image datasets: {len(train_dataset)}')
    print(f'\tVal image datasets: {len(val_dataset)}')
    # Patches
    train_dataset = images_to_patches(train_dataset, patch_size=args['patch_size'])
    val_dataset = images_to_patches(val_dataset, patch_size=args['patch_size'])
    print(f'\tTrain patch datasets: {len(train_dataset)}')
    print(f'\tVal patch datasets: {len(val_dataset)}')

    # DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=6)
    val_dataloader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=6)
    print(f'\tTrain batch number: {len(train_dataloader)}')
    print(f'\tVal batch number: {len(val_dataloader)}')

    # Noise list
    train_noises = args['train_noise_interval'] # [0, 75, 15]
    val_noises = args['val_noise_interval'] # [0, 60, 30]
    train_noises = list(range(train_noises[0], train_noises[1], train_noises[2]))
    val_noises = list(range(val_noises[0], val_noises[1], val_noises[2]))
    print(f'\tTrain noise internal: {train_noises}')
    print(f'\tVal noise internal: {val_noises}')
    print('\n')

    # Model & Optim
    model = FFDNet(is_gray=args['is_gray'])
    model.apply(utils.weights_init_kaiming)
    if args['cuda']:
        model = model.cuda()
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])

    print('> Start training...')
    for epoch_idx in range(args['epoches']):
        # Train
        loss_idx = 0
        train_losses = 0
        model.train()

        start_time = time.time()
        for batch_idx, batch_data in tqdm(enumerate(train_dataloader)):
            #print(batch_data.size)
            # According to internal, add noise
            for int_noise_sigma in train_noises:
                
                noise_sigma = int_noise_sigma / 255
                new_images = utils.add_batch_noise(batch_data, noise_sigma)
                noise_sigma = torch.FloatTensor(np.array([noise_sigma for idx in range(new_images.shape[0])]))
                new_images = Variable(new_images)
                noise_sigma = Variable(noise_sigma)
                if args['cuda']:
                    new_images = new_images.cuda()
                    noise_sigma = noise_sigma.cuda()

                # Predict
                images_pred = model(new_images, noise_sigma)
                train_loss = loss_fn(images_pred, batch_data.to(images_pred.device))
                train_losses += train_loss
                loss_idx += 1

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                # Log Progress
                stop_time = time.time()
                all_num = len(train_dataloader) * len(train_noises)
                done_num = batch_idx * len(train_noises) + train_noises.index(int_noise_sigma) + 1
                rest_time = int((stop_time - start_time) / done_num * (all_num - done_num))
                percent = int(done_num / all_num * 100)
                print(f'\rEpoch: {epoch_idx + 1} / {args["epoches"]}, ' +
                      f'Batch: {batch_idx + 1} / {len(train_dataloader)}, ' +
                      f'Noise_Sigma: {int_noise_sigma} / {train_noises[-1]}, ' +
                      f'Train_Loss: {train_loss}, ' +
                      f'=> {rest_time}s, {percent}%', end='')

        train_losses /= loss_idx
        print(f', Avg_Train_Loss: {train_losses}, All: {int(stop_time - start_time)}s')
        

if __name__ == '__main__':
    train()
