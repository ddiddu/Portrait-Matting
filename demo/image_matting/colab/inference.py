"""
Example:
    python -m demo.image_matting.colab.inference --data-path=baidu_V1 --ckpt-path=pretrained/modnet_webcam_portrait_matting.ckpt
"""

import os
import argparse
import numpy as np
from PIL import Image
import random
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.models.modnet import MODNet


def unify_img(im):
    # unify image channels to 3
    im = np.asarray(im)
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]
    return im


def process_image(im):
    # define hyper-parameters
    ref_size = 512
    
    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    im = unify_img(im)

    # convert image to PyTorch tensor
    im = Image.fromarray(im)
    im = im_transform(im)

    # add mini-batch dim
    im = im[None, :, :, :]

    # resize image for input
    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w
    
    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

    return im, im_h, im_w


def combined_display(image, matte, background):
    # obtain predicted foreground
    image = unify_img(image)
    matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
    foreground = image * matte + background * (1 - matte)
    # foreground = image * matte + np.full(image.shape, 0) * (1 - matte)

    foreground = foreground.reshape(image.shape)
    foreground = Image.fromarray(np.uint8(foreground))
    return foreground


if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='path of data')
    parser.add_argument('--ckpt-path', type=str, help='path of pre-trained MODNet')
    args = parser.parse_args()

    # check input arguments
    if not os.path.exists(args.ckpt_path):
        print('Cannot find ckpt path: {0}'.format(args.ckpt_path))
        exit()

    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=True)
    modnet = nn.DataParallel(modnet).cuda()
    state_dict = torch.load(args.ckpt_path)
    modnet.load_state_dict(state_dict)
    modnet.eval()

    data_paths = args.data_path.split('/')
    bgd_path = os.path.join('Data', 'places365')
    place_names = os.path.join(bgd_path, 'places.txt')
    with open(place_names) as data:
        bgd_names = data.read().splitlines()

    for data_path in data_paths:
        # inference images
        data_path = os.path.join('Data', args.data_path)
        input_path = os.path.join(data_path, 'input')
        input_bgd_path = os.path.join(data_path, 'input_aug_bgd')
        output_path = os.path.join(data_path, 'output')
        file_names = os.path.join(input_path, 'train.txt')
        with open(file_names) as data:
            im_names = data.read().splitlines()

        for im_name in im_names:
#             print('Process image: {0}'.format(im_name))

            # read image
            im_path = os.path.join(input_path, im_name)
            # supervisely portrait
            if not os.path.isfile(im_path):
                im_path = im_path.split('.')[0] + '.jpeg'
            image = Image.open(im_path)
            im, im_h, im_w = process_image(image)

            # inference
            _, _, matte = modnet(im.cuda(), True) # cuda(): RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

            # resize and save matte
            matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
            matte = matte[0][0].data.cpu().numpy()
            matte = Image.fromarray(np.uint8(matte * 255))
            
            # ground truth
            matte_name = im_name.split('.')[0] + '.jpg'
            matte.save(os.path.join(output_path, matte_name))

#             for idx in range (10):
#                 bgd_lists = random.choice(bgd_names)
#                 bgd_dir = os.path.join(bgd_path, bgd_lists)
#                 bgd = random.choice(os.listdir(bgd_dir))
                
#                 background = cv2.resize(cv2.imread(os.path.join(bgd_dir, bgd)), (im_w, im_h))
#                 foreground = combined_display(image, matte, background)
#                 foreground_name = im_name.split('.')[0] + '_' + str(idx) + '.jpg'
#                 foreground.save(os.path.join(input_bgd_path, foreground_name))

#                 matte_name = im_name.split('.')[0] + '_' + str(idx) + '.jpg'
#                 matte.save(os.path.join(output_path, matte_name))
