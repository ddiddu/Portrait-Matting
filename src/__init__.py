"""
Example:
    python -m src.__init__ --backbone=mnv2 --train-data=SPS --ckpt-path=pretrained/modnet_webcam_portrait_matting.ckpt --output-path=pretrained/tmp.ckpt --consistency=True --bgd=False
"""

import os
import argparse
from PIL import Image

import torch
from src.models.modnet import MODNet
from src.models.modnet_st import MODNet_ST
from src.models.modnet_mnv3 import MODNet_MNV3
from src.trainer import fine_tuning_iter, soc_mnv3_iter
from demo.image_matting.colab.inference import process_image

import torch.utils.data as data
import pytorch_model_summary

import warnings
warnings.filterwarnings("ignore")

from tensorboardX import SummaryWriter
writer = SummaryWriter()


# Create your dataloader
class Human(data.Dataset):
    def __init__(self, train_data, image_names, consistency=False, bgd=False):
        self.consistency = consistency
        self.bgd = bgd
        self.im_names = []
        with open(image_names) as data:
            images = data.read().splitlines()
            for im in images:
                if im.split('/')[0] in train_data:
                    if consistency == True:
                        im = os.path.join(im.split('/')[0], im.split('/')[1] + '_aug' , im.split('/')[2])
                    if bgd == True:
                        im = os.path.join(im.split('/')[0], im.split('/')[1] + '_bgd' , im.split('/')[2])
                    self.im_names.append(im)

    def __getitem__(self, idx):
        im_name = self.im_names[idx]

        # read image
        im_path = os.path.join('Data', im_name)
        im = Image.open(im_path)

        im, _, _ = process_image(im)
        
        mt_name = im_name.split('/')[-1].split('.')[0] + '.jpg'
        output_path = os.path.join('Data', im_name.split('/')[0], 'output')
        if self.consistency == True:
            output_path = output_path + '_aug'
        if self.bgd == True:
            output_path = output_path + '_bgd'
        mt_path = os.path.join(output_path, mt_name)
        
        if not os.path.isfile(mt_path):
            mt_path = mt_path.split('.')[0] + '.png'
        
        if not os.path.isfile(mt_path):
            mt_path = mt_path.split('.')[0] + '.png'
            
        mt = Image.open(mt_path)
        
        mt, _, _ = process_image(mt)

        return im, mt
    
    def __len__(self):
        return len(self.im_names)

if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, help='backbone model of MODNet')
    parser.add_argument('--train-data', type=str, help='path of data')
    parser.add_argument('--ckpt-path', type=str, help='path of pre-trained MODNet')
    parser.add_argument('--output-path', type=str, help='path of trained checkpoint')
    parser.add_argument('--consistency', type=str, help='use of consistency constraint loss')
    parser.add_argument('--bgd', type=str, help='use of background removal')
    args = parser.parse_args()

    # check input arguments
    if not os.path.exists(args.ckpt_path):
        print('Cannot find ckpt path: {0}'.format(args.ckpt_path))
        exit()
    
    bs = 16         # batch size
    lr = 0.01       # learning rate
    if args.backbone == "mnv2":
        lr = 0.00001
    epochs = 10     # total epochs
    workers = 4     # number of data laoding workers

    if args.backbone == "mnv2":
        modnet = torch.nn.DataParallel(MODNet_ST(backbone_pretrained=True)).cuda()
        
        # Load pretrained ckpt
        state_dict = torch.load(args.ckpt_path)
        """ Load state dict partial """
        new_model_dict = modnet.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in new_model_dict}
        new_model_dict.update(pretrained_dict)
        modnet.load_state_dict(new_model_dict)

    else:
        modnet = torch.nn.DataParallel(MODNet_MNV3(backbone_pretrained=True)).cuda()
        
        backup_modnet = torch.nn.DataParallel(MODNet(backbone_pretrained=True)).cuda()
        # Load pretrained ckpt
        backup_modnet.load_state_dict(torch.load(args.ckpt_path))
        
    train_data = args.train_data.split('/')
    optimizer = torch.optim.Adam(modnet.parameters(), lr=lr, betas=(0.9, 0.99))
    
    # set training set and validation set
    # if args.consistency == 'True':
    #     input_path = input_path + '_aug'
    #     output_path = output_path + '_aug'
    # if args.bgd == 'True':
    #     input_path = input_path + '_bgd'
    #     output_path = output_path + '_bgd'
    dataset_train = Human(train_data=train_data, 
                          image_names = os.path.join('Data', 'train.txt'),)
                        #   consistency = args.consistency)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=bs, shuffle=True, num_workers=workers)

    dataset_val = Human(train_data=train_data,
                        image_names = os.path.join('Data', 'val.txt'))
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False)
    print('image number in training: %d, validation: %d' % (len(dataset_train), len(dataset_val)))

    for epoch in range(0, epochs):
        losses = 0
        for idx, (images, mattes) in enumerate(dataloader_train): # torch.Size([bs, 1, 3, 512, 512]) torch.Size([bs, 1, 3, 512, 512])
            batch_losses = 0
            for batch_idx in range(len(images)): # bs
                image = images[batch_idx]
                matte = mattes[batch_idx]
                if args.backbone == "mnv2":
                    batch_losses = batch_losses + fine_tuning_iter(modnet, optimizer, image, matte.cuda(), validation=False)
                else:
                    soc_semantic_loss, soc_detail_loss, soc_matte_loss = soc_mnv3_iter(modnet, backup_modnet, optimizer, image, matte.cuda(), validation=False)
                    batch_losses = batch_losses + soc_semantic_loss + soc_detail_loss + soc_matte_loss
            batch_loss = batch_losses / (batch_idx+1)
            losses = losses + batch_loss
        loss = losses / (idx+1)

        # validation
        if len(dataset_val) != 0:
            mse_losses = 0
            mad_losses = 0
            for idx, ([image], [matte]) in enumerate(dataloader_val):
                if args.backbone == "mnv2":
                    mse, mad = fine_tuning_iter(modnet, optimizer, image, matte.cuda(), validation=True)
                else:
                    mse, mad = soc_mnv3_iter(modnet, backup_modnet, optimizer, image, matte.cuda(), validation=True)
                mse_losses = mse_losses + mse
                mad_losses = mad_losses + mad
            mse_loss = mse_losses / (idx+1)
            mad_loss = mad_losses / (idx+1)
        else:
            mse_loss = 0
            mad_loss = 0

        print('Epoch %d/%d - loss: %.3f - mse_loss: %.5f - mad_loss: %.5f' % (epoch+1, epochs, loss, mse_loss, mad_loss))
        writer.add_scalar("train loss", loss/100, epoch)
        writer.add_scalars('val loss', {'mse': mse_loss, 'mad': mad_loss}, epoch)

        if epoch == 0:
            min_mse_loss = mse_loss   
            min_mad_loss = mad_loss

        if mse_loss <= min_mse_loss:
            min_mse_loss = mse_loss
            mse_path = args.output_path.split('.')[0] + '_mse.ckpt'
            torch.save(modnet.state_dict(), mse_path)
            print('Epoch 000%d: saving model to %s' % (epoch+1, mse_path))

        if mad_loss <= min_mad_loss:
            min_mad_loss = mad_loss
            mad_path = args.output_path.split('.')[0] + '_mad.ckpt'
            torch.save(modnet.state_dict(), mad_path)
            print('Epoch 000%d: saving model to %s' % (epoch+1, mad_path))

        # Initial learning rate is 0.01 and is multiplied by 0.1 after every 10 epochs
        if args.backbone == "mnv3":
            if epoch == 10:
                lr = lr * 0.1

    print(pytorch_model_summary.summary(modnet, torch.zeros(1, 3, 224, 224), show_input=True))
    print(pytorch_model_summary.summary(backup_modnet, torch.zeros(1, 3, 224, 224), show_input=True))

    writer.flush()
    writer.close()
