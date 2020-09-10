import os, sys, argparse
import numpy as np
import torch.backends.cudnn as cudnn

from albumentations import (
    Compose, ShiftScaleRotate, PadIfNeeded, RandomCrop, Normalize, HorizontalFlip, VerticalFlip,
    ElasticTransform, RandomBrightnessContrast, CenterCrop, CropNonEmptyMaskIfExists, RandomResizedCrop,
    GaussNoise, Rotate
)
from albumentations.pytorch import ToTensorV2

sys.path.append('./src')
from losses import BootstrapDiceLoss
from deeplab import DeepLabV3
from data import SegmentationData
from torch.optim import AdamW
from torch.utils.data import DataLoader
from train_utils import Trainer

def parse_args():
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Train a model on given dataset')
    parser.add_argument('data_path', type=str, metavar='data_path', help='Path containing tiff label images')
    parser.add_argument('save_path', type=str, metavar='save_path', help='Path to save model state')
    parser.add_argument('-n', type=int, metavar='n', help='number of segmentation classes')
    parser.add_argument('-lr', type=float, metavar='lr', help='learning rate')
    parser.add_argument('-wd', type=float, metavar='wd', help='weight decay')
    parser.add_argument('-iters', type=int, metavar='iters', help='total training iterations')
    parser.add_argument('-bsz', type=int, metavar='bsz', help='batch size')
    parser.add_argument('-p', type=float, metavar='p', help='dropout p')
    parser.add_argument('-beta', type=float, metavar='beta', help='dice loss beta value [0, 1]')
    
    parser.add_argument('--resnet_arch', dest='resnet_arch', type=str, metavar='resnet_arch', 
                        default='resnet34', choices=['resnet18', 'resnet34', 'resnet50'], help='resnet model to use as deeplab backbone')
    parser.add_argument('--ft_layer', dest='ft_layer', type=str, metavar='ft_layer', 
                        default='none', help='resnet layers to finetune: none, layer4, layer3, layer2, layer1, all')
    parser.add_argument('--resume', type=str, metavar='p', help='Path to model state for resuming training', 
                        default=None)
    
    args = vars(parser.parse_args())
    
    return args

def snakemake_args():
    params = vars(snakemake.params)
    params['data_path'] = snakemake.input[0]
    params['save_path'] = snakemake.output[0]
    del params['_names']
    
    return params

if __name__ == "__main__":
    #determine if the script is being run by Snakemake
    if 'snakemake' in globals():
        args = snakemake_args()
    else:
        args = parse_args()

    data_path = args['data_path']
    save_path = args['save_path']
    num_classes = args['n']
    lr = args['lr']
    wd = args['wd']
    iters = args['iters']
    bsz = args['bsz']
    p = args['p']
    beta = args['beta']
    resnet_arch = args['resnet_arch']
    finetune_layer = args['ft_layer']
    resume = args['resume']

    #set the transforms to use, we're using imagenet
    #pretrained models, so we want to use the default
    #normalization parameters and copy our input image
    #channels such that there are 3 (RGB)
    tfs = Compose([
        PadIfNeeded(min_height=224, min_width=224),
        CropNonEmptyMaskIfExists(224, 224, p=0.8),
        RandomResizedCrop(224, 224, (0.6, 1.0)),
        RandomBrightnessContrast(),
        #Rotate(),
        HorizontalFlip(),
        VerticalFlip(),
        GaussNoise(var_limit=(5.0, 20.0), p=0.25),
        Normalize(),
        ToTensorV2()
    ])
    
    eval_tfs = Compose([
        PadIfNeeded(min_height=224, min_width=224),
        CenterCrop(224, 224),
        Normalize(),
        ToTensorV2()
    ])

    #create pytorch datasets
    #use the batch size that we defined in the arguments
    #but the batch size should not be greater than the number 
    #of images in the trn_data
    trn_data = SegmentationData(data_path, tfs)
    bsz = min(bsz, len(trn_data))
    train = DataLoader(trn_data, batch_size=bsz, shuffle=True, pin_memory=True, num_workers=8, drop_last=True)
    
    #make validation loader, if validation data exists
    if os.path.exists(data_path.replace('train2d', 'valid2d')) and 'train2d' in data_path:
        val_data = SegmentationData(data_path.replace('train2d', 'valid2d'), tfs)
        valid = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=True, num_workers=8, drop_last=False)
    else:
        valid = None

    #create deeplab model
    model = DeepLabV3(num_classes, resnet_arch, pretrained=True, drop_p=p)
    
    #freeze weights in resnet based on the given finetune layer
    encoder_groups = [mod[1] for mod in model.encoder.resnet.named_children()]
    if finetune_layer != 'none':
        layer_index = {'all': 0, 'layer1': 4, 'layer2': 5, 'layer3': 6, 'layer4': 7}
        start_layer = layer_index[finetune_layer]

        #always finetune from the start layer to the last layer in the resnet
        for group in encoder_groups[start_layer:]:
            for param in group.parameters():
                param.requires_grad = True
                
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Froze resnet weights before {finetune_layer}. Using model with {params} trainable parameters!')

    #use AdamW optimizer and bootstrapped dice loss
    optim = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = BootstrapDiceLoss(beta=beta)

    #train the model with one cycle policy
    cudnn.benchmark = True
    trainer = Trainer(model, optim, criterion, train, valid)
    
    resume = None if resume == '' else resume
    trainer.train_one_cycle(iters, lr, iters // 10, save_path=save_path, resume=resume)
