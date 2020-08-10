import os
import argparse
import numpy as np
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm

def parse_args():
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Create dataset for nn experimentation')
    parser.add_argument('impath', type=str, metavar='impaths', help='Path to an image file')
    parser.add_argument('mskpath', type=str, metavar='mskpaths', help='Path to labelmap file')
    parser.add_argument('dataset_name', type=str, metavar='dataset_name', help='Name prefix of datasets')
    parser.add_argument('save_path', type=str, metavar='save_path', help='Path to save the slices')
    parser.add_argument('-a', '--axes', dest='axes', type=int, metavar='axes', nargs='+', default=[0, 1, 2],
                        help='Volume axes along which to slice (0-yz, 1-xz, 2-xy)')
    parser.add_argument('-s', '--spacing', dest='spacing', type=int, metavar='spacing', default=1,
                        help='Spacing between image slices')
    

    args = vars(parser.parse_args())
    
    return args

def snakemake_args():
    params = vars(snakemake.params)
    params['impath'] = snakemake.input[0]
    params['mskpath'] = snakemake.input[1]
    params['save_path'] = snakemake.output[0]
    
    return params

#args = parse_args()
args = snakemake_args()

#read in the parser arguments
impath = args['impath']
mskpath = args['mskpath']
dataset_name = args['dataset_name']
save_path = args['save_path']
axes = args['axes']
spacing = args['spacing']

#gather the image and mask files
impaths = np.sort(glob(impath + dataset_name + '*'))
mskpaths = np.sort(glob(mskpath + dataset_name + '*sem_seg*'))
print(dataset_name, impaths, mskpaths)
    
#create images and masks directories in save_path
#if they do not already exist
if not os.path.exists(save_path):
    os.mkdir(save_path)
    
if not os.path.exists(save_path + '/images/'):
    os.mkdir(save_path + '/images/')
    
if not os.path.exists(save_path + '/masks/'):
    os.mkdir(save_path + '/masks/')
    
#load the image and labelmap volumes
for impath, mskpath in zip(impaths, mskpaths):
    image = sitk.ReadImage(impath)
    labelmap = sitk.ReadImage(mskpath) > 0
    #labelmap = sitk.BinaryDilate(labelmap, 2)

    #establish a filename prefix from the impath
    exp_name = impath.split('/')[-1].split('.')[0]

    #loop over the axes and save slices
    for axis in axes:
        #get the axis dimension and get evenly spaced slice indices
        slice_indices = np.arange(0, image.GetSize()[axis] - 1, spacing, dtype=np.long)
        for idx in slice_indices:
            idx = int(idx)
            slice_name = '_'.join([exp_name, str(axis), str(idx) + '.tiff'])
            #don't save anything if the slice already exists
            if os.path.exists(save_path + 'images/' + slice_name):
                continue

            if axis == 0:
                image_slice = image[idx]
                labelmap_slice = labelmap[idx]
            elif axis == 1:
                image_slice = image[:, idx]
                labelmap_slice = labelmap[:, idx]
            else:
                image_slice = image[:, :, idx]
                labelmap_slice = labelmap[:, :, idx]

            #binarize the labelmap slice
            #labelmap_slice = labelmap_slice > 200
            
            #if the labelmap is empty, don't save anything
            #if sitk.GetArrayFromImage(labelmap_slice).max() == 1:
            sitk.WriteImage(image_slice, save_path + '/images/' + slice_name)
            sitk.WriteImage(labelmap_slice, save_path + '/masks/' + slice_name)
