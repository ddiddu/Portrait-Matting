"""
Example:
    python -m Data.dataset --data-path=SPS --ckpt-path=pretrained/modnet_webcam_portrait_matting.ckpt
"""

import os
import argparse
import numpy as np
from PIL import Image
import cv2
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modnet import MODNet
from Data.data_aug import data_aug_flip
from Data.data_aug import data_aug_blur, data_aug_color, data_aug_noise
from demo.image_matting.colab.inference import process_image


def augment(image, mask):
    img_aug_ori = np.array(image)

    img_aug = Image.fromarray(cv2.cvtColor(img_aug_ori, cv2.COLOR_BGR2RGB))
    img_aug = data_aug_color(img_aug)
    img_aug = np.asarray(img_aug)
    img_aug = data_aug_blur(img_aug)
    img_aug = data_aug_noise(img_aug)
    img_aug = np.float32(img_aug[:,:,::-1]) # BGR, like cv2.imread

    img_aug, mask_aug, _ = data_aug_flip(img_aug, mask)

    return img_aug, mask_aug


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

    for data_path in data_paths:
        # inference images
        data_path = os.path.join('Data', args.data_path)
        input_path = os.path.join(data_path, 'input')
        aug_path = os.path.join(data_path, 'input_aug')
        output_path = os.path.join(data_path, 'output_aug')
        file_names = os.path.join(input_path, 'train.txt')
        with open(file_names) as data:
            im_names = data.read().splitlines()

        for im_name in im_names:
#             print('Process image: {0}'.format(im_name))
            # read image
            im_path = os.path.join(input_path, im_name)
#             aug_name = im_name.split('.')[0] + '.jpg'
#             shutil.copyfile(im_path, os.path.join(aug_path, aug_name))
            image = Image.open(im_path)
            im, im_h, im_w = process_image(image)

            # inference
            _, _, matte = modnet(im.cuda(), True) # cuda(): RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

            # resize and save matte
            matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
            matte = matte[0][0].data.cpu().numpy()
            mask = Image.fromarray(np.uint8(matte * 255))
#             matte_name = im_name.split('.')[0] + '.jpg'
#             mask.save(os.path.join(output_path, matte_name))

            for idx in range (10):
                img_aug, mask_aug = augment(image, matte)
                
                aug_name = im_name.split('.')[0] + '_' + str(idx) + '.jpg'
                Image.fromarray(np.uint8(img_aug)).save(os.path.join(aug_path, aug_name))
                
                # ground truth
                mask = Image.fromarray(np.uint8(mask_aug * 255))
                matte_name = im_name.split('.')[0] + '_' + str(idx) + '.jpg'
                mask.save(os.path.join(output_path, matte_name))
