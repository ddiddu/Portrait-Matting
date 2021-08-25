"""
Export ONNX model of MODNet with:
    input shape: (batch_size, 3, height, width)
    output shape: (batch_size, 1, height, width)  

Arguments:
    --ckpt-path: path of the checkpoint that will be converted
    --output-path: path for saving the ONNX model

Example:
    python -m onnx.export_onnx --ckpt-path=pretrained\mnv2_vol1_SPS_matte_aug_all_val_mse.ckpt --output-path=model\mnv2_vol1_SPS_matte_aug_all_val_mse.onnx
"""

import os
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable

from . import modnet_onnx
from . import modnet_mnv3_onnx


if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=True, help='path of the checkpoint that will be converted')
    parser.add_argument('--output-path', type=str, required=True, help='path for saving the ONNX model')
    args = parser.parse_args()

    # check input arguments
    if not os.path.exists(args.ckpt_path):
        print('Cannot find checkpoint path: {0}'.format(args.ckpt_path))
        exit()

    # define model & load checkpoint
    modnet = modnet_onnx.MODNet(backbone_pretrained=True, only_semantic=False)
    # modnet = modnet_mnv3_onnx.MODNet(backbone_pretrained=True)
    modnet = nn.DataParallel(modnet).cuda()
    state_dict = torch.load(args.ckpt_path)

    """ Load state dict partial """
    new_model_dict = modnet.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    
    modnet.load_state_dict(new_model_dict)
    modnet.eval()

    # prepare dummy_input
    batch_size = 1
    height = 512
    width = 512
    dummy_input = Variable(torch.randn(batch_size, 3, height, width)).cuda()

    # export to onnx model
    torch.onnx.export(
        modnet.module, dummy_input, args.output_path, export_params = True, opset_version=11,
        input_names = ['input'], output_names = ['output'], 
        dynamic_axes = {'input': {0:'batch_size', 2:'height', 3:'width'}, 'output': {0: 'batch_size', 2: 'height', 3: 'width'}})
