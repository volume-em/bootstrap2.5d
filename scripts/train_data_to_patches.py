import os
import argparse
import numpy as np
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm

def parse_args():
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Creates images from orthogonal 2d cross sections of image and mask volumes in imdir and mskdir.')
    parser.add_argument('imdir', type=str, metavar='impath', help='Path to directory containing image volumes')
    parser.add_argument('mskdir', type=str, metavar='mskpath', help='Path to directory containing mask volumes')
    parser.add_argument('save_path', type=str, metavar='save_path', help='Path to save the patches')
    parser.add_argument('-a', '--axes', dest='axes', type=int, metavar='axes', nargs='+', default=[0, 1, 2],
                        help='Volume axes along which to slice (0-yz, 1-xz, 2-xy)')
    parser.add_argument('-s', '--spacing', dest='spacing', type=int, metavar='spacing', default=1,
                        help='Spacing between image slices')
    
    args = vars(parser.parse_args())
    return args

def snakemake_args():
    params = vars(snakemake.params)
    params['imdir'] = snakemake.input[0]
    params['mskdir'] = snakemake.input[1]
    params['save_path'] = snakemake.output[0]
    del params['_names']
    
    return params

if __name__ == "__main__":
    #determine if the script is being run by Snakemake
    if 'snakemake' in globals():
        args = snakemake_args()
    else:
        args = parse_args()

    #read in the parser arguments
    imdir = args['imdir']
    mskdir = args['mskdir']
    save_path = args['save_path']
    axes = args['axes']
    spacing = args['spacing']

    #create images and masks directories in save_path
    #if they do not already exist
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    images_path = os.path.join(save_path, 'images')
    if not os.path.exists(images_path):
        os.mkdir(images_path)
    
    masks_path = os.path.join(save_path, 'masks')
    if not os.path.exists(masks_path):
        os.mkdir(masks_path)
        
    #glob the lists of image and mask volumes
    impaths = np.sort(glob(os.path.join(imdir, '*')))
    mskpaths = np.sort(glob(os.path.join(mskdir, '*')))
    
    #make sure all the image and mask volumes have a partner
    #with the same name
    assert(len(impaths) == len(mskpaths)), "imdir and mskdir must have the same number of volumes!"
    for impath, mskpath in zip(impaths, mskpaths):
        assert(impath.split('/')[-1] == mskpath.split('/')[-1]), \
        f'Image-mask pairs must have the same names. Got {impath} and {mskpath}'
        
        #load the image and labelmap volumes
        image = sitk.ReadImage(impath)
        labelmap = sitk.ReadImage(mskpath)

        #establish a filename prefix from the impath
        exp_name = '.'.join(impath.split('/')[-1].split('.')[:-1])

        #loop over the axes and save slices
        for axis in axes:
            #get the axis dimension and get evenly spaced slice indices
            slice_indices = np.arange(0, image.GetSize()[axis] - 1, spacing, dtype=np.long)
            for idx in slice_indices:
                idx = int(idx)
                slice_name = '_'.join([exp_name, str(axis), str(idx) + '.tiff'])

                if axis == 0:
                    image_slice = image[idx]
                    labelmap_slice = labelmap[idx]
                elif axis == 1:
                    image_slice = image[:, idx]
                    labelmap_slice = labelmap[:, idx]
                else:
                    image_slice = image[:, :, idx]
                    labelmap_slice = labelmap[:, :, idx]

                sitk.WriteImage(image_slice, os.path.join(images_patch, slice_name))
                sitk.WriteImage(labelmap_slice, os.path.join(masks_path, slice_name))