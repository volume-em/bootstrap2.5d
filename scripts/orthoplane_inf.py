import numpy as np
import os, sys, argparse, warnings, cv2
import torch
import torch.nn as nn
import SimpleITK as sitk
from skimage import io
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader

from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2

sys.path.append('../')
from deeplab import DeepLabV3

def parse_args():
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Run orthoplane inference on volumes in a given directory')
    parser.add_argument('imdir', type=str, metavar='imvol', help='Directory containing image volumes')
    parser.add_argument('weight_path', type=str, metavar='weight_path', help='Path to model state file')
    parser.add_argument('save_dir', type=str, metavar='save_dir', help='Directory to save segmentation result')
    parser.add_argument('-n', type=int, metavar='n', help='Number of segmentation classes')
    parser.add_argument('--axes', dest='axes', type=int, metavar='axes', nargs='+', default=[0, 1, 2],
                        help='Volume axes along which to predict (0-yx, 1-zx, 2-zy)')
    parser.add_argument('--threshold', type=float, metavar='threshold', default=0.5,
                        help='Prediction confidence of threshold')
    parser.add_argument('--resnet_arch', dest='resnet_arch', type=str, metavar='resnet_arch', 
                        default='resnet34', choices=['resnet18', 'resnet34', 'resnet50'], help='resnet model to use as deeplab backbone')
    
    args = vars(parser.parse_args())
    
    return args

def snakemake_args():
    params = vars(snakemake.params)
    params['imdir'] = snakemake.input[0]
    params['weight_path'] = snakemake.input[1]
    params['save_dir'] = snakemake.output[0]
    del params['_names']
    
    return params

if __name__ == "__main__":
    #determine if the script is being run by Snakemake
    if 'snakemake' in globals():
        args = snakemake_args()
    else:
        args = parse_args()
        
    #parse the arguments
    imdir = args['imdir']
    weight_path = args['weight_path']
    save_dir = args['save_dir']
    num_classes = args['n']
    axes = args['axes']
    threshold = args['threshold']
    resnet_arch = args['resnet_arch']
    
    #glob the volumes in the directory
    impaths = glob(os.path.join(imdir, '*'))
    print(f'Found {len(impaths)} for inference in {imdir}')
    
    #create the save path if it doesn't exist
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    #setup the model and augmentations
    eval_tfs = Compose([
        Normalize(),
        ToTensorV2()
    ])
    
    #create model
    model = DeepLabV3(num_classes, resnet_arch, pretrained=False, drop_p=0)
    
    #load the model file and get the weights
    state = torch.load(weight_path, map_location='cpu')['model']
    model.load_state_dict(state)
    print(f'Loaded weights from {weight_path}')
    
    #set scaling factor based on number of axes
    scaling = int(255 / len(axes))
    
    #loop through the found image volumes
    for impath in impaths:
        print(f'Evaluating volume {impath}...')

        #load the image volume
        orig_vol = sitk.ReadImage(impath)
        im_vol = sitk.GetArrayFromImage(orig_vol)
        print(f'Volume size {im_vol.shape}')

        prediction_volume = np.zeros((num_classes, *im_vol.shape), dtype=np.uint8)

        #main loop for orthoplane inference
        for ax in axes:
            print(f'Predicting over axis {ax}...')
            stack = np.split(im_vol, im_vol.shape[ax], axis=ax)
            for index, image in enumerate(stack):
                #convert image to 3 channel grayscale
                image = cv2.cvtColor(np.squeeze(image), cv2.COLOR_GRAY2RGB)

                #apply the augmentations and add the batch dimension
                image = eval_tfs(image=image)['image'].unsqueeze(0)

                #load image to gpu
                image = image.cuda() #(1, 1, H, W)

                #get the image size and calculate the required padding
                #such that image shape is divisble by 32
                h, w = image.size()[2:]
                pad_bottom = 32 - h % 32 if h % 32 != 0 else 0
                pad_right = 32 - w % 32 if w % 32 != 0 else 0
                #reflection pad to bottom and right only
                image = nn.ReflectionPad2d((0, pad_right, 0, pad_bottom))(image)

                #evaluate the image
                with torch.no_grad():
                    prediction = model.eval()(image)
                    prediction = prediction[..., :h, :w] #crop off the padding

                    #apply either sigmoid or softmax based on number of classes
                    if num_classes == 1:
                        prediction = nn.Sigmoid()(prediction) #(1, 1, H, W)
                    else:
                        prediction = nn.Softmax(dim=1)(prediction) #(1, NC, H, W)

                    #convert to uint8 numpy array
                    #use scaling such that we can add the results together
                    #adding now with uint8 is substantially more memory efficient
                    #than using float32, and only loses an insignificant amount of precision
                    prediction = (prediction.squeeze(0).detach().cpu().numpy() * scaling).astype(np.uint8) #(NC, H, W)
                    if ax == 0:
                        prediction_volume[:, index] += prediction
                    elif ax == 1:
                        prediction_volume[:, :, index] += prediction
                    else:
                        prediction_volume[:, :, :, index] += prediction

        #if we're working with a single class
        #use the threshold, otherwise use the
        #argmax of softmax
        threshold = int(255 * threshold)
        if num_classes == 1:
            prediction_volume = (prediction_volume > threshold).astype(np.uint8)[0]
        else:
            prediction_volume = np.argmax(prediction_volume, axis=0).astype(np.uint8)

        #convert from numpy array to simpleITK image
        #copy information from the original volume so that
        #spacing and origin will be correct
        prediction_volume = sitk.GetImageFromArray(prediction_volume)
        prediction_volume.CopyInformation(orig_vol)
        
        #save the result with the same name and type as the image volume
        volume_name = impath.split('/')[-1]
        save_path = os.path.join(save_dir, volume_name)
        sitk.WriteImage(prediction_volume, save_path)
        print(f'Saved orthoplane inference result to {save_path}')