import cv2, os
from skimage.io import imread
import numpy as np
import torchvision.transforms as tf
import torchvision.transforms.functional as TF
import torch
from PIL import Image as pil
from torch.utils.data import DataLoader, Dataset
from albumentations.augmentations import functional as F
from albumentations import DualTransform, ImageOnlyTransform, to_tuple

class MitoData(Dataset):
    """
    A regular Pytorch dataset that loads tiff files and applies image transforms.
    
    impath: Path containing tiff images
    
    mskpath: Path containing tiff masks. If this dataset it to be used for inference, then set
    mskpath to None. Calls to __getitem__ will only return im and fname if None.
    
    fnames: List of filenames included in the dataset, for example, there should be a
    different list for training and validation datasets. This list can be generated with
    next(os.walk(impath))[2] and then choose indices of files in dataset. Note that
    image and mask filenames were generated to be the same, so only 1 list is necessary
    for loading both image and mask files.
    
    tfs: Pytorch transforms, if None then no transforms are applied to input
    
    
    Returns:
    -----------
    A dictionary containing the keys fname, im, msk where im and msk are transfomed pytorch tensors
    """
    
    def __init__(self, data_dir, tfs=None, pseudo_dataset=None, gray_channels=3, data_fraction=1.0, binarize_mask=False):
        super(MitoData, self).__init__()
        self.data_dir = data_dir
        self.impath = os.path.join(data_dir, 'images')
        self.mskpath = os.path.join(data_dir, 'masks')
        self.binarize_mask = binarize_mask
        
        self.fnames = np.array(next(os.walk(self.impath))[2])
        print(f'Found {len(self.fnames)} images in {self.impath}')
        
        #indices = []
        #for ix, fn in enumerate(self.fnames):
        #    if 6 in np.unique(imread(self.mskpath + '/' + fn)):
        #        indices.append(ix)    
        #self.fnames = self.fnames[indices][:16]
        #print(f'Filtered image to {len(self.fnames)} with class 6')
        
        #if data_fraction < 1.0:
        #    num_images = int(len(self.fnames) * data_fraction)
        #    num_images = max(num_images, 16)
        #    self.fnames = np.random.choice(self.fnames, num_images, replace=False)
        #    print(f'Using {num_images} random images')
        
        self.tfs = tfs
        self.pseudo_dataset = pseudo_dataset
        self.gray_channels = gray_channels
        self.length = len(self.fnames)
        
    def __len__(self):
        return self.length
    
    def get_pseudo_data(self, idx):
        return self.pseudo_dataset.get_data(idx)
    
    def get_data(self, idx):
        #np.random.seed(None)
        f = self.fnames[idx]
        image = imread(self.impath + '/' + f)
            
        image = image.astype(np.uint8)
        assert(image.dtype == np.uint8), "Image not uint8!!"

        if self.gray_channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            #add an empty channel dim
            image = image[..., None]
        
        #load the mask image
        mask = imread(self.mskpath + '/' + f)
        mask = mask.astype(np.uint8)
        
        if self.binarize_mask:
            mask = (mask > 0).astype(np.uint8)

        if self.tfs is not None:
            transformed = self.tfs(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            
        return f, image, mask
    
    def __getitem__(self, idx):
        #get the filename from list
        if self.pseudo_dataset is None:
            f, image, mask = self.get_data(idx)
            
        else:
            if np.random.random() > 0.3:
                f, image, mask = self.get_data(idx)
            else:
                f, image, mask = self.get_pseudo_data(idx % len(self.pseudo_dataset))
                
        return {'fname': f,
                'im': image,
                'msk': mask.float()
               }
        
class FactorResize(DualTransform):

    def __init__(self, resize_factor, always_apply=False, p=1.0):
        super(FactorResize, self).__init__(always_apply, p)
        self.rf = resize_factor

    def apply(self, img, **params):
        h, w = img.shape[:2]
        nh = int(h / self.rf) * self.rf
        nw = int(w / self.rf) * self.rf
        
        #set the interpolator and return
        interpolation = cv2.INTER_NEAREST
        return F.resize(img, nh, nw, interpolation)