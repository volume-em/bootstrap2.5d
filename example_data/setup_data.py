import os, argparse, subprocess
import SimpleITK as sitk
import numpy as np
from glob import glob

def parse_args():
    # Setup the argument parser
    parser = argparse.ArgumentParser(description='Process the UroCell data for model training and testing')
    parser.add_argument('data_dir', type=str, metavar='imvol', help='Directory containing image and mask data volumes')
    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":
    # Parse argument
    args = parse_args()
    data_dir = args['data_dir']
    
    # Download the UroCell dataset
    script_path = os.path.join(os.path.dirname(__file__), 'download_example.sh')
    command = f'bash {script_path} {data_dir}'
    subprocess.call(command.split(' '))
    
    # Convert images from float type to uint8
    impaths = glob(os.path.join(data_dir, 'train/images/*.nii.gz')) + glob(os.path.join(data_dir, 'target/images/*.nii.gz'))
    for impath in impaths:
        sitk.WriteImage(sitk.Cast(sitk.ReadImage(impath), sitk.sitkUInt8), impath)
    
    print('Creating mask volumes...')
    # Get paths of the mito and lyso labelmap volumes
    lysopaths = np.sort(glob(os.path.join(data_dir, 'train/lyso/*.nii.gz')) + glob(os.path.join(data_dir, 'target/lyso/*.nii.gz')))
    mitopaths = np.sort(glob(os.path.join(data_dir, 'train/mito/*.nii.gz')) + glob(os.path.join(data_dir, 'target/mito/*.nii.gz')))

    for lp, mp in zip(lysopaths, mitopaths):
        assert(lp.replace('/lyso/', '/mito/') == mp), "Lyso and mito label volumes are not aligned!"

        # Load the volumes
        lyso = sitk.ReadImage(lp)
        mito = sitk.ReadImage(mp)

        # Add them together into a single label volume such that 1 == lyso & 2 == mito
        labelmap = lyso + 2 * mito

        # Make sure the datatype is uint8
        labelmap = sitk.Cast(labelmap, sitk.sitkUInt8)

        # Save the result
        sitk.WriteImage(labelmap, lp.replace('lyso/', 'masks/'))
        
    print('Done!')