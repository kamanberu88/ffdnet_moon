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
        'patch_size': 32,
        'train_noise_interval': [0, 75, 15],
        'val_noise_interval': [0, 60, 30],
        'batch_size': 256,
        'epoches': 80,
        'val_epoch': 5,
        'learning_rate': 1e-3,
        'save_checkpoints': 5,
        'model_path': './models/',
        'use_gpu': True
    }
    args['cuda'] = args['use_gpu'] and torch.cuda.is_available()

    print('> Loading dataset...')
    # Images
    train_dataset = load_images(is_train=True, is_gray=args['is_gray'], base_path=args['train_path'])
    val_dataset = load_images(is_train=False, is_gray=args['is_gray'], base_path=args['train_path'])
    print(f'\tTrain image datasets: {len(train_dataset)}')
    print(f'\tVal image datasets: {len(val_dataset)}')
 
    # Patches
    train_dataset = images_to_patches(train_dataset, patch_size=args['patch_size'])
    val_dataset = images_to_patches(val_dataset, patch_size=args['patch_size'])
    print(f'\tTrain patch datasets: {len(train_dataset)}')
    print(f'\tVal patch datasets: {len(val_dataset)}')

    print('> Building model...')
    model = FFDNet(args['is_gray'])
    if args['cuda']:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
    criterion = nn.MSELoss()

    print('> Start training...')
    for epoch in range(1, args['epoches'] + 1):
        #print(f'Epoch {epoch} / {args['epoches']}')
        print(f"Epoch {epoch} / {args['epoches']}")
        start_time = time.time()
        # Training
        model.train()
        epoch_loss = 0
        for i in range(len(train_dataset) // args['batch_size']):
            inputs = train_dataset[i * args['batch_size']:(i + 1) * args['batch_size']]
            inputs = utils.add_batch_noise(inputs, args['train_noise_interval'])
            inputs = torch.from_numpy(inputs).float()
            targets = torch.from_numpy(train_dataset[i * args['batch_size']:(i + 1) * args['batch_size']]).float()
            if args['cuda']:
                inputs = inputs.cuda()
                targets = targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f'Training Loss: {epoch_loss / len(train_dataset)}')
        print(f'Time: {time.time() - start_time} sec')

        # Validation
        if epoch % args['val_epoch'] == 0:
            model.eval()
            avg_psnr = 0
            with torch.no_grad():
                for i in range(len(val_dataset) // args['batch_size']):
                    inputs = val_dataset[i * args['batch_size']:(i + 1) * args['batch_size']]
                    inputs = utils.add_noise(inputs, args['val_noise_interval'])
                    inputs = torch.from_numpy(inputs).float()
                    targets = torch.from_numpy(val_dataset[i * args['batch_size']:(i + 1) * args['batch_size']]).float()
                    if args['cuda']:
                        inputs = inputs.cuda()
                        targets = targets.cuda()

                    outputs = model(inputs)
                    mse = F.mse_loss(outputs, targets)
                    psnr = 10 * torch.log10(1 / mse.item())
                    avg_psnr += psnr
            print(f'Average PSNR: {avg_psnr / len(val_dataset)} dB')

        # Save checkpoints
        if epoch % args['save_checkpoints'] == 0:
            torch.save(model.state_dict(), f"{args['model_path']}/model_epoch_{epoch}.pth")

    print('> Training completed!')
    torch.save(model.state_dict(), f"{args['model_path']}/model_final.pth")

if __name__ == '__main__':
    train()
