"""

This script loads an image file and model weights and performs inference.

"""

import os, sys, argparse, mlflow
import torch
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from glob import glob
from albumentations import Compose, PadIfNeeded, Normalize
from albumentations.pytorch import ToTensorV2
#from torchvision.models import resnet34

sys.path.append('/home/conradrw/nbs/moco_official/')
from moco.resnet import resnet50 as moco_resnet50

sys.path.append('/home/conradrw/nbs/mitonet/model2d/')
from deeplab import DeepLabV3
from data import MitoData, FactorResize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BatchLoader:
    def __init__(self, im_vol, plane_index, batch_size, tfs):
        self.im_vol = im_vol
        self.bsz = batch_size
        self.plane_index = plane_index
        self.tfs = tfs
        self.sl_index = 0
    
    def get_next_slice(self):
        if self.plane_index == 0:
            next_slice = sitk.GetArrayFromImage(self.im_vol[self.sl_index, :, :])
        elif self.plane_index == 1:
            next_slice = sitk.GetArrayFromImage(self.im_vol[:, self.sl_index, :])
        else:
            next_slice = sitk.GetArrayFromImage(self.im_vol[:, :, self.sl_index])
            
        self.sl_index += 1
        
        #return slice with a channel dim at last axis
        return next_slice[..., None]

    def __len__(self):
        size = self.im_vol.GetSize()[self.plane_index]
        if size % self.bsz == 0:
            return size // self.bsz
        else:
            return size // self.bsz + 1
    
    def transform_batch(self, batch):
        #loop through all the slices and apply
        #transforms to each, return the list
        #of transformed images
        if self.tfs is not None:
            tfs_batch = [self.tfs(image=sl)['image'] for sl in batch]
        else:
            tfs_batch = batch

        return tfs_batch
    
    def get_next_batch(self):
        sl_limit = self.im_vol.GetSize()[self.plane_index]
        insert_idx = self.sl_index
        
        batch = [self.get_next_slice() for _ in range(self.bsz) if self.sl_index < sl_limit]
        batch = self.transform_batch(batch)
        #stack images into a batch B x (N, H, W) --> (B, N, H, W)
        batch = torch.stack(batch, dim=0)
        
        return insert_idx, self.plane_index, batch

def resize_volume(volume, factor=16):
    #resize the given volume such that all dimensions
    #are divisible by the given factor
    #get the original size
    xi, yi, zi = im_vol.GetSize()
    
    #change to the closest product of 16
    xf = int(xi / factor) * factor
    yf = int(yi / factor) * factor
    zf = int(zi / factor) * factor
    sizef = [xf, yf, zf]
    
    #create an empty volume of new size and same type
    reference = sitk.Image(sizef, volume.GetPixelID())
    
    #the origin and direction should be the same, but 
    #the spacing must change
    #xspi, yspi, zspi = volume.GetSpacing()
    #spacingf = [xspi * (xi / xf), yspi * (yi / yf), zspi * (zi / zf)]
    reference.SetOrigin(volume.GetOrigin())
    reference.SetDirection(volume.GetDirection())
    #reference.SetSpacing(spacingf)
    reference.SetSpacing(volume.GetSpacing())
    
    return sitk.Resample(volume, reference)
    
def make_empty_labelmap(im_vol):
    label_vol = sitk.Image(im_vol.GetSize(), 1)
    label_vol.SetOrigin(im_vol.GetOrigin())
    label_vol.SetSpacing(im_vol.GetSpacing())
    
    return label_vol

def update_labelmap(label_vol, batch_predictions, sl_index, plane_index):
    #batch_predictions are a torch tensor (B, 1, H, W)
    #first strip the channel dim and convert to sitk images
    #squeeze the channel dim only
    batch_predictions = batch_predictions.squeeze(1)
    pred_images = [sitk.GetImageFromArray(p) for p in batch_predictions]
    
    #make the pred_images into a 3d stack
    pred_stack = sitk.JoinSeries(pred_images)
    
    #set the destinationIndex
    if plane_index == 0:
        pred_stack = sitk.PermuteAxes(pred_stack, [2, 0, 1])
        dest_idx = [sl_index, 0, 0]
    elif plane_index == 1:
        pred_stack = sitk.PermuteAxes(pred_stack, [0, 2, 1])
        dest_idx = [0, sl_index, 0]
    else:
        dest_idx = [0, 0, sl_index]
    
    #set the correct spacing
    pred_stack.SetSpacing(label_vol.GetSpacing())
    
    #paste the prediction stack at the given destination_index
    return sitk.Paste(label_vol, pred_stack, pred_stack.GetSize(), destinationIndex=dest_idx)
    
def parse_args():
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Train a model on given dataset')
    parser.add_argument('impath', type=str, metavar='data_path', help='Path to image volume')
    parser.add_argument('weight_path', type=str, metavar='weight_path', help='Path to model state file')
    parser.add_argument('save_path', type=str, metavar='save_path', help='Path to save segmentation result')
    args = vars(parser.parse_args())
    
    return args

def snakemake_args():
    params = {} #vars(snakemake.params)
    params['impath'] = snakemake.input[0]
    params['weight_path'] = snakemake.input[1]
    params['save_path'] = snakemake.output[0]
    #params['save_path2'] = snakemake.output[1]
    #params['save_path3'] = snakemake.output[2]
    
    return params

#args = parse_args()
args = snakemake_args()
    
#extract the arguments
impath = args['impath']
weight_path = args['weight_path']
save_path = args['save_path']
#save_paths = [args['save_path1'], args['save_path2'], args['save_path3']]
#thresholds = args['thresholds']

out_path = '/'.join(save_path.split('/')[:-1])

if not os.path.exists(out_path):
    os.mkdir(out_path)


#load the image volume
im_vol = sitk.ReadImage(impath)
im_vol_resized = resize_volume(im_vol, 16)
print('Image volume resized to {}'.format(im_vol_resized.GetSize()))

#create 2 empty label volumes in the
#same physical space as im_vol
#one will be for the final prediction
#the other will be for storing intermediates
label_vol = make_empty_labelmap(im_vol_resized)
label_vol_tmp = make_empty_labelmap(im_vol_resized)

#save slices from the volume as tiffs and get the
#filenames that were used (already sorted by function)
#fnames = volume2slices(im_vol, slice_path)

#next we want to determine a batch size of images
#to feed through the network at each step, a good
#rough estimate is based on some testing I did on 224x224
#size images to see how many would fit in memory
slice_w, slice_h = im_vol.GetSize()[0], im_vol.GetSize()[-1]
bsz = int(((224 * 224) / (slice_w * slice_h)) * 256)

#set the transforms to use, we're using imagenet
#pretrained models, so we want to use the default
#normalization parameters and copy our input image
#channels such that there are 3 (RGB)
tfs = Compose([
    Normalize(mean=[0.66527], std=[0.12484]),
    ToTensorV2()
])

#load the weight file and check if it's really weights or a state file
state = torch.load(weight_path)
state_dict = state['state_dict']
run_id = state['run_id']

#create model based on given model name
resnet = moco_resnet50()
model = DeepLabV3(resnet, 1).to(device)
model.load_state_dict(state_dict)

#create the data loaders
loader_xy = BatchLoader(im_vol_resized, 2, bsz, tfs)
loader_xz = BatchLoader(im_vol_resized, 1, bsz, tfs)
loader_yz = BatchLoader(im_vol_resized, 0, bsz, tfs)

with torch.no_grad():
    #loop over the batches from each dataloader
    print('predicting xy...')
    for _ in range(len(loader_xy)):
        #move batch to device
        insert_idx, plane_index, batch = loader_xy.get_next_batch()
        batch = batch.to(device)

        #run inference (B, 3, H, W) --> (B, 1, H, W)
        preds = model.eval()(batch).detach().cpu()
        preds = torch.nn.Sigmoid()(preds).numpy()
        preds = (preds * 85).astype(np.uint8)

        #insert the predictions into the label vol
        #and return the updated version
        label_vol_tmp = update_labelmap(label_vol_tmp, preds, insert_idx, plane_index)

    #add the temp label_vol to master
    label_vol_tmp.SetDirection(label_vol.GetDirection())
    label_vol += label_vol_tmp
    label_vol.SetDirection(im_vol.GetDirection())

    print('predicting xz...')
    for _ in range(len(loader_xz)):
        #move batch to device
        insert_idx, plane_index, batch = loader_xz.get_next_batch()
        batch = batch.to(device)

        #run inference (B, 3, H, W) --> (B, 1, H, W)
        preds = model.eval()(batch).detach().cpu()
        preds = torch.nn.Sigmoid()(preds).numpy()
        preds = (preds * 85).astype(np.uint8)

        #insert the predictions into the label vol
        #and return the updated version
        label_vol_tmp = update_labelmap(label_vol_tmp, preds, insert_idx, plane_index)

    #add the temp label_vol to master
    label_vol_tmp.SetDirection(label_vol.GetDirection())
    label_vol += label_vol_tmp
    label_vol.SetDirection(im_vol.GetDirection())

    print('predicting yz...')
    for _ in range(len(loader_yz)):
        #move batch to device
        insert_idx, plane_index, batch = loader_yz.get_next_batch()
        batch = batch.to(device)

        #run inference (B, 3, H, W) --> (B, 1, H, W)
        preds = model.eval()(batch).detach().cpu()
        preds = torch.nn.Sigmoid()(preds).numpy()
        preds = (preds * 85).astype(np.uint8)

        #insert the predictions into the label vol
        #and return the updated version
        label_vol_tmp = update_labelmap(label_vol_tmp, preds, insert_idx, plane_index)

    #add the temp label_vol to master
    label_vol_tmp.SetDirection(label_vol.GetDirection())
    label_vol += label_vol_tmp 
    label_vol.SetDirection(im_vol.GetDirection())


#the label_vol should be good now, let's save the result and finish
#set the directions to match
#establish a filename prefix from the impath
label_vol = sitk.Resample(label_vol, im_vol)
sitk.WriteImage(label_vol, save_path)

#apply the thresholding value that was given
#apply the opening operation to remove small objects (smaller than 5**3 voxels)
#for thr, save_path in zip(thresholds, save_paths):
#thr_int = int(thr * 255)
#print('Thr {}, Label vol max {}'.format(thr_int, sitk.GetArrayFromImage(label_vol).max()))
#label_vol_post = label_vol > thr_int
#print('Post', sitk.GetArrayFromImage(label_vol_post).max())
#label_vol_post = sitk.BinaryOpeningByReconstruction(label_vol_post, [5, 5, 5])
#print('Post opening', sitk.GetArrayFromImage(label_vol_post).max())
#label_vol_post = sitk.BinaryErode(label_vol_post, 1)
#print('Post erode', sitk.GetArrayFromImage(label_vol_post).max())
#sitk.WriteImage(label_vol_post, save_path)
