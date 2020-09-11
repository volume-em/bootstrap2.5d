import os
import argparse
import numpy as np
import SimpleITK as sitk
from glob import glob

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
    parser.add_argument('-eval_frac', 'eval_frac', dest='eval_frac', type=float, metavar='eval_frac', default=0.15,
                        help='Fraction of total images and masks to use for validation dataset [0, 1]')
    parser.add_argument('-eval_path', 'eval_path', dest='eval_path', type=str, metavar='eval_path', default="",
                        help='Directory in which to save validation images and masks')
    
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
    eval_frac = args['eval_frac']
    eval_path = args['eval_path']
        
    #create save_path
    #if is does not already exist
    train_path = save_path
    os.mkdir(os.path.join(train_path, 'images'))
    os.mkdir(os.path.join(train_path, 'masks'))
    
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
        
        assert(image.GetPixelID() == 1), f"{impath} is not 8-bit unsigned integer!"
        assert(image.GetSize() == labelmap.GetSize()), f"{impath} and {mskpath} volumes are not the same (x, y, z) shape!"

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

                sitk.WriteImage(image_slice, os.path.join(os.path.join(train_path, 'images'), slice_name))
                sitk.WriteImage(labelmap_slice, os.path.join(os.path.join(train_path, 'masks'), slice_name))
                
    #create validation directories
    if eval_frac > 0 and eval_path != "":
        os.mkdir(eval_path)
        os.mkdir(os.path.join(eval_path, 'images'))
        os.mkdir(os.path.join(eval_path, 'masks'))

        #get the list of images and masks in train directory
        train_images = np.sort(glob(os.path.join(train_path, 'images/*.tiff')))
        train_masks = np.sort(glob(os.path.join(train_path, 'masks/*.tiff')))

        #create a mask to randomly select images
        mask = np.random.random(len(train_images)) < eval_frac
        for imp, mskp in zip(train_images[mask], train_masks[mask]):
            os.rename(imp, imp.replace('train', 'valid'))
            os.rename(mskp, mskp.replace('train', 'valid'))

        print(f'Saved {mask.sum()} images for validation dataset.')