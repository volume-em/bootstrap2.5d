import os

#Location of the vol2images.py, train.py and orthoplane_inf.py scripts
#By default they are in the same directory as the Snakefile
SCRIPT_PATH = ""

#Directory containing training and target data volumes
DATA_PATH = "data/"

#Directory in which to save pytorch models
MODEL_PATH = "models/"

#vol2images.py script slices a volume along xy, yz, xz planes and
#creates 2d images for each cross section; these cross sections
#are saved in the paths below
TRAIN_PATCHES_PATH = DATA_PATH + "train2d/" 

#Directory for saving some cross sections for validation instead of training
VALID_PATCHES_PATH = DATA_PATH + "valid2d/"

#Directory containing training image volumes
TRAINING_IMDIR = DATA_PATH + "train/images/"

#Directory containing training label volumes
TRAINING_MSKDIR = DATA_PATH + "train/masks/" 

#Directory to save 2d noisy ground truth masks and 2d cross sections
#of the target image volume
TARGET_PATCHES_PATH = DATA_PATH + "target2d/" 

#Directory containing target image volumes
TARGET_IMDIR = DATA_PATH + "target/images/" 

#Directory to save supervised prediction volumes (output of Step 1)
TARGET_PRED_SUPER_DIR = DATA_PATH + "target/super_preds/" 

#Directory to save weakly supervised prediction volumes (output of Step 2)
TARGET_PRED_WEAKSUPER_DIR = DATA_PATH + "target/weaksuper_preds/"

#Resnet architecture to use as the DeepLab encoder
#choice of resnet18, resnet34, or resnet50
RESNET_ARCH = "resnet34"

#Number of classes in the segmentation mask
N_CLASSES = 1

rule all:
    input:
        TARGET_PRED_SUPER_DIR,
        TARGET_PRED_WEAKSUPER_DIR

        
#--------------------------------------------
# SUPERVISED Training and Inference -- Step 1
#--------------------------------------------

rule train_data_to_patches:
    input:
        TRAINING_IMDIR,
        TRAINING_MSKDIR
    params:
        axes = [0, 1, 2],
        spacing = 1,
        eval_frac = 0.15,
        eval_path = VALID_PATCHES_PATH
    output:
        directory(TRAIN_PATCHES_PATH),
        directory(VALID_PATCHES_PATH)
    script:
        SCRIPT_PATH + "vol2images.py"
        
rule train_supervised:
    input:
        TRAIN_PATCHES_PATH
    params:
        n = N_CLASSES, #number of segmentation classes in the mask
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
        
        
#---------------------------------------------------
# WEAKLY SUPERVISED Training and Inference -- Step 2
#---------------------------------------------------

rule target_data_to_patches:
    input:
        TARGET_IMDIR,
        TARGET_PRED_SUPER_DIR
    params:
        axes = [0, 1, 2],
        spacing = 1,
        eval_frac = 0.0, #validation data is meaningless in this step, because there isn't a real ground truth
        eval_path = ""
    output:
        directory(TARGET_PATCHES_PATH)
    script:
        SCRIPT_PATH + "vol2images.py"
        
rule train_weakly_supervised:
    input:
        TARGET_PATCHES_PATH
    params:
        n = N_CLASSES, #number of segmentation classes in the mask
        lr = 3e-3, #maximum learning rate in OneCycle policy
        wd = 0.1, #weight decay
        iters = 1000, #total training iterations
        bsz = 64, #batch size, no smaller than 16
        p = 0.5, #dropout probability
        beta = 0.8, #with bootstrapping
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
