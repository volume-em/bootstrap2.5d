import numpy as np
from PIL import Image as pil
from tqdm import tqdm_notebook as tqdm

def np3d2tiff(nrrd_array, save_path, source='roi1'):
    
    planes = ['yz', 'xz', 'xy']
    
    count = 0
    for p in planes:
        for i,s in enumerate(nrrd_array):
            im = pil.fromarray(s)
            fname = '{}_slice{}_{}.tiff'.format(source, i, p)
            im.save(save_path + fname)
            count += 1
            
        #permute axes and repeat    
        nrrd_array = np.transpose(nrrd_array, [1, 2, 0])
        
    return "{} images saved to {}".format(count, save_path)

def np2tiff(numpy_array, save_path, source='cellA'):
    
    planes = ['yz', 'xz', 'xy']
    
    num_im = np.sum(numpy_array.shape)
    prefix_form = '%0{}d'.format(np.ceil(np.log10(num_im)))
    
    count = 0
    for p in planes:
        for i,s in tqdm(enumerate(numpy_array)):
            
            im = pil.fromarray(s)
            prefix = prefix_form % i
            fname = '{}_{}_{}.tiff'.format(prefix, source, p)
            im.save(save_path + fname)
            count += 1
            
        #permute axes and repeat    
        numpy_array = np.transpose(numpy_array, [1, 2, 0])
        
    return "{} images saved to {}".format(count, save_path)


def rle_encoding(x):
    '''
    x: torch tensor of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def stitch_crop(img, size):
    """
    Crop the given Pytorch Image (channels, h, w) into chunks as close to the given size as possible without overlapping slices
    and while keeping each slice the same size.
    
    Size should be in dimensions of (height, width)
    """
    
    h, w = img.size()[1:]
    
    #first calculate ideal factors for width and height
    wif = round(w / size[1]) + 1
    hif = round(h / size[0]) + 1
    
    #now calculate the factors for img width and height and only keep the one closest to wif, hif
    wnf = np.array([i for i in range(1, int(np.sqrt(w))) if w % i == 0])
    wnf = wnf[np.argmin(np.abs(wnf - wif))]
    hnf = np.array([i for i in range(1, int(np.sqrt(h))) if h % i == 0])
    hnf = hnf[np.argmin(np.abs(hnf - hif))]
    
    crop_w = w / wnf
    crop_h = h / hnf
    
    #a list for storing our cropped images
    crops = []
    
    #the algorithm for cropping is to start in the upper left corner of
    #the original image and slide across the width cutting wnf slices
    #once we reach the end of the width, we return to x=0 and increment
    #our y coordinate down 1 full crop_h distance
    y = 0
    #proceed along the height
    for hix in range(hnf):
        y = hix * crop_h
        x = 0
        #proceed along the width
        for wix in range(wnf):
            left, upper = int(x), int(y)
            right, lower = int(left + crop_w), int(upper + crop_h)
            
            #now crop the image and append to the crops list
            crops.append(img[:, upper:lower, left:right])
            
            #lastly, increment the x (left) coordinate
            x += crop_w
    
    return crops, (wnf, hnf)