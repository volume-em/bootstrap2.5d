import os

SCRIPT_PATH = "scripts/"

DATA_PATH = "data/"
MODEL_PATH = "models/"

TRAIN_PATCHES_PATH = DATA_PATH + "train2d/" #location to save 2d training images
TRAINING_IMDIR = DATA_PATH + "train/images/" #directory containing training image volumes
TRAINING_MSKDIR = DATA_PATH + "train/masks/" #directory containing training label volumes

TARGET_PATCHES_PATH = DATA_PATH + "target2d/" #location to save 2d noisy ground truth images
TARGET_IMDIR = DATA_PATH + "target/images/" #directory containing target image volumes
TARGET_PRED_SUPER_DIR = DATA_PATH + "target/super_preds/" #directory to save supervised prediction volumes
TARGET_PRED_WEAKSUPER_DIR = DATA_PATH + "target/weaksuper_preds/" #directory to save weakly supervised prediction volumes

RESNET_ARCH = "resnet34"

rule all:
    input:
        TARGET_PRED_SUPER_DIR,
        TARGET_PRED_WEAKSUPER_DIR

        
#----------------------------------
# SUPERVISED
#----------------------------------

"""

take the volumes in imdir and mskdir and save the 2d cross sections to PATCHES_PATH
this script/rule assumes a directory structure like:
data/
    images/
        volume1.tiff
        volume2.tiff
    masks/
        volume1.tiff
        volume2.tiff
imdir = data/images/ and mskdir = data/masks/

"""
rule train_data_to_patches:
    input:
        TRAINING_IMDIR,
        TRAINING_MSKDIR
    params:
        axes = [0, 1, 2],
        spacing = 1,
        eval_frac = 0.15
    output:
        directory(TRAIN_PATCHES_PATH)
    script:
        SCRIPT_PATH + "vol2images.py"
        
rule train_supervised:
    input:
        TRAIN_PATCHES_PATH
    params:
        n = 1, #number of segmentation classes in the mask
        lr = 3e-3, #maximum learning rate in OneCycle policy
        wd = 0.1, #weight decay
        iters = 1000, #total training iterations
        bsz = 64, #batch size, no smaller than 16
        p = 0.5, #dropout probability
        beta = 1, #no bootstrapping
        resnet_arch = RESNET_ARCH, #resnet18, resnet34, or resnet50
        ft_layer = "layer4", #all, layer1, layer2, layer3, layer4, or none.
        resume = "" #resuming is not compatible with scripts run by Snakemake
    output:
        os.path.join(MODEL_PATH, "supervised.pth")
    script:
        SCRIPT_PATH + "train.py"
        
rule orthoplane_inf_supervised:
    input:
        TARGET_IMDIR,
        os.path.join(MODEL_PATH, "supervised.pth")
    params:
        n = 1, #number of segmentation classes in the mask
        axes = [0, 1, 2],
        threshold = 0.1, 
        resnet_arch = RESNET_ARCH #resnet18, resnet34, or resnet50
    output:
        directory(TARGET_PRED_SUPER_DIR),
    script:
        SCRIPT_PATH + "orthoplane_inf.py"
        
        
#----------------------------------
# WEAKLY SUPERVISED
#----------------------------------

rule target_data_to_patches:
    input:
        TARGET_IMDIR,
        TARGET_PRED_SUPER_DIR
    params:
        axes = [0, 1, 2],
        spacing = 1,
        eval_frac = 0.0 #validation data is meaningless in this step, because there isn't a real ground truth
    output:
        directory(TARGET_PATCHES_PATH)
    script:
        SCRIPT_PATH + "vol2images.py"
        
rule train_weakly_supervised:
    input:
        TARGET_PATCHES_PATH
    params:
        n = 1, #number of segmentation classes in the mask
        lr = 3e-3, #maximum learning rate in OneCycle policy
        wd = 0.1, #weight decay
        iters = 1000, #total training iterations
        bsz = 64, #batch size, no smaller than 16
        p = 0.5, #dropout probability
        beta = 1, #no bootstrapping
        resnet_arch = RESNET_ARCH, #resnet18, resnet34, or resnet50
        ft_layer = "layer4", #all, layer1, layer2, layer3, layer4, or none.
        resume = "" #resuming is not compatible with scripts run by Snakemake
    output:
        os.path.join(MODEL_PATH, "weakly_supervised.pth")
    script:
        SCRIPT_PATH + "train.py"
        
rule orthoplane_inf_weakly_supervised:
    input:
        TARGET_IMDIR,
        os.path.join(MODEL_PATH, "weakly_supervised.pth")
    params:
        n = 1, #number of segmentation classes in the mask
        axes = [0, 1, 2],
        threshold = 0.5, 
        resnet_arch = RESNET_ARCH #resnet18, resnet34, or resnet50
    output:
        directory(TARGET_PRED_WEAKSUPER_DIR),
    script:
        SCRIPT_PATH + "orthoplane_inf.py"