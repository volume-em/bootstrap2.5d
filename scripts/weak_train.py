import os, sys, argparse, mlflow
import numpy as np
#from torchvision.models import resnet34
import torch
import torch.backends.cudnn as cudnn

from albumentations import (
    Compose, ShiftScaleRotate, PadIfNeeded, RandomCrop, Normalize, HorizontalFlip, VerticalFlip,
    ElasticTransform, RandomBrightnessContrast, CenterCrop, CropNonEmptyMaskIfExists, RandomResizedCrop,
    GaussNoise, RandomRotate90
)

from albumentations.pytorch import ToTensorV2

sys.path.append('/home/conradrw/nbs/moco_official/')
from moco.resnet import resnet50 as moco_resnet50

sys.path.append('/home/conradrw/nbs/mitonet/model2d/')
from deeplab import MobileDeepLab, DeepLabV3
from data import MitoData

sys.path.append('/home/conradrw/nbs/mitonet/utils/')
from metrics import IoU, Dice, BalancedAccuracy, ComposeMetrics
from losses import HardDiceLoss2
from torch.optim import AdamW
from torch.utils.data import DataLoader
from train_utils import Trainer

MOCO_WEIGHTS = "/data/IASEM/conradrw/models/MoCoV2/current.pth.tar"

def parse_args():
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Train a model on given dataset')
    parser.add_argument('data_path', type=str, metavar='data_path', help='Path containing tiff label images')
    parser.add_argument('save_path', type=str, metavar='save_path', help='Path to save model state')
    parser.add_argument('experiment', type=str, metavar='save_path', help='mlflow experiment name')
    parser.add_argument('-lr', type=str, metavar='lr', help='learning rate')
    parser.add_argument('-wd', type=str, metavar='wd', help='weight decay')
    parser.add_argument('-iters', type=str, metavar='iters', help='total training iterations')
    parser.add_argument('-bsz', type=str, metavar='bsz', help='batch size')
    parser.add_argument('-p', type=str, metavar='p', help='dropout p')
    args = vars(parser.parse_args())
    
    return args

def snakemake_args():
    params = vars(snakemake.params)
    params['data_path'] = snakemake.input[0]
    params['save_path'] = snakemake.output[0]
    
    return params

#args = parse_args()
args = snakemake_args()

data_path = args['data_path']
save_path = args['save_path']
lr = args['lr']
wd = args['wd']
iters = args['iters']
bsz = args['bsz']
p = args['p']
experiment = args['experiment']
beta = float(args['beta'])

#set the transforms to use, we're using imagenet
#pretrained models, so we want to use the default
#normalization parameters and copy our input image
#channels such that there are 3 (RGB)
tfs = Compose([
    PadIfNeeded(min_height=224, min_width=224),
    #CropNonEmptyMaskIfExists(224, 224),
    RandomResizedCrop(224, 224, (0.5, 1.0)),
    #RandomBrightnessContrast(),
    RandomRotate90(),
    HorizontalFlip(),
    VerticalFlip(),
    GaussNoise(var_limit=(5.0, 20.0), p=0.25),
    Normalize(mean=[0.66527], std=[0.12484]),
    ToTensorV2()
])

#create pytorch datasets
trn_data = MitoData(data_path, tfs, gray_channels=1)

#use the batch size that we defined in the arguments
#but the batch size should not be greater than the number 
#of images in the trn_fnames
train = DataLoader(trn_data, batch_size=bsz, shuffle=True, pin_memory=True, num_workers=8, drop_last=True)

#create model
resnet = moco_resnet50()

#load the state dict for mocov2
checkpoint = torch.load(MOCO_WEIGHTS, map_location="cpu")

# rename moco pre-trained keys
state_dict = checkpoint['state_dict']
for k in list(state_dict.keys()):
    # retain only encoder_q up to before the embedding layer
    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
        # remove prefix
        state_dict[k[len("module.encoder_q."):]] = state_dict[k]

    # delete renamed or unused k
    del state_dict[k]


msg = resnet.load_state_dict(state_dict, strict=False)
model = DeepLabV3(resnet, 1, drop_p=p)

#best to use differential learning rates, so define the param_groups
#and optimizer with best hyperparameters
param_groups = [
    {'params': model.backbone.parameters()},
    {'params': model.aspp.parameters()},
    {'params': model.decoder.parameters()}
]

optim = AdamW(param_groups, lr=lr, weight_decay=wd)
loss = HardDiceLoss2(beta=beta)

#define metrics and class names
cn = ['Mito']
md = {'IoU': IoU(), 'Dice': Dice(), 'BalAcc': BalancedAccuracy()}
metrics = ComposeMetrics(md, cn)

logging_params = {
    'experiment_name': experiment, 'max_lr': str(lr), 'batch_size': str(bsz), 'lr_schedule': 'one_cycle',
    'dropout': str(p), 'optimizer': 'AdamW', 'loss': 'BootstrapSoftDice', 'architecture': 'DeepLabV3+,MoCo50',
    'augmentations': 'random_resized_crops, flips, brightness, rotate90, noise', 
    'pretraining': 'moco, unfreeze layer 4', 'iterations': str(iters), 'weight_path': save_path,
    'data_path': data_path, 'beta': beta
}

trainer = Trainer(model, optim, loss, train, val_data=None, metrics=metrics, logging=logging_params)

for param in trainer.model.backbone[7].parameters():
    param.requires_grad = True
    
cudnn.benchmark = True

trainer.train_one_cycle(iters, lr, iters // 50, save_path=save_path)
#trainer.train_step_decay(50, [30, 40], 0.1, save_path=save_path)
