# Bootstrap 2.5d
Code for the paper: [Enforcing Prediction Consistency Across Orthogonal Planes Significantly Improves Segmentation of FIB-SEM Image Volumes by 2D Neural Networks.](https://www.cambridge.org/core/journals/microscopy-and-microanalysis/article/enforcing-prediction-consistency-across-orthogonal-planes-significantly-improves-segmentation-of-fibsem-image-volumes-by-2d-neural-networks/97314A3AF09213E4E97491D95BF03C1B)

## Running in Google Colab

The easiest way to get started is to run the code on Google Colab.

Anyone with a Google account can use Colab for free, albeit with some [resource limits](https://research.google.com/colaboratory/faq.html#resource-limits). Without a Google account the code can still be viewed and downloaded, but a sign-in is required to edit and execute. The button below will open a new tab with the notebook opened in Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/volume-em/bootstrap2.5d/blob/master/colab_notebook.ipynb)

## Running on your own machine

### Dependencies

By default, this code is designed to run with an Nvidia GPU; it will not work on CPU-only machines without modification. The code was tested on Ubuntu 18.04 with Python 3.8.

We recommend using conda for managing the python environment and installing all necessary packages to run this code. For detailed instructions see [here](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html). 

For basic Miniconda installation on Ubuntu run:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Then create a conda environment from the environment.yml file.
```
conda env create -f environment.yml
```
To activate the environment, use:
```
conda activate boot
```

### Using this code

#### Cloning and directory creation

The easiest way to run the entire pipeline is by using [Snakemake](https://snakemake.readthedocs.io/en/stable/). Start by cloning this repo and making some directories.

```
git clone https://github.com/volume-em/bootstrap2.5d
cd bootstrap2.5d
mkdir data
mkdir models

mkdir data/train/
mkdir data/train/images/
mkdir data/train/masks/

mkdir data/target/
mkdir data/target/images/
```

#### Load datasets

Next, move training image and labelmap volumes into the ```data/train/images/``` and ```data/train/masks/``` directories, respectively. Note that the image and labelmap volume files must have exactly the same names and must have 8-bit unsigned voxels. Lastly, move a target image volume for segmentation into the ```data/target/images/``` directory. Multiple volumes can be in each directory, training and inference will be applied to each as a group. See [here](https://simpleitk.readthedocs.io/en/master/IO.html) for a list of supported file formats.

The directory structure will look something like this:
```
bootstrap2.5d\
    models\
    data\
        train\
            images\
                train_volume1.tiff
                train_volume2.tiff
            masks\
                train_volume1.tiff
                train_volume2.tiff
        target\
            images\
                target_volume1.tiff
                target_volume2.tiff
    
```

#### Running Snakemake

Once the directories are setup, the last step is to run snakemake with the given Snakefile.
```
snakemake
```

#### Ouptut files

The Snakefile has multiple outputs. The most important are the pytorch model state dicts and predicted segmentation volumes. The model state dicts will appear in the ```models``` directory. One will be called ```supervised.pth``` and the other called ```weakly_supervised.pth``` (these names should be obvious after reading the paper). The predicted segmentation volumes will appear in the ```target``` directory under ```super_preds``` and ```weaksuper_preds```.

#### Clear Snakemake Outputs and Rerun

To rerun the Snakefile with different data or hyperparameters, it may be necessary to remove all previously generated snakemake results. This can be done manually if only certain output need to be modified. For example, to repredict ```weaksuper_preds``` only, just remove that directory:
```
rm -r data/target/weaksuper_preds/
```

To remove all outputs from the entire pipeline run:
```
snakemake --delete-all-output
```

### Modifying the Snakefile

The directory structure and pipeline hyperparameters can easily be modified by editing the Snakefile. Detailed comments in the file explain the purpose of each parameter.

## Citation
If you use this code, please cite our paper:

Conrad, R., Lee, H., & Narayan, K. (2020). Enforcing Prediction Consistency Across Orthogonal Planes Significantly Improves Segmentation of FIB-SEM Image Volumes by 2D Neural Networks. Microscopy and Microanalysis, 1-4. doi:10.1017/S143192762002053X

## Acknowledgements

This project was funded in part with Federal funds from the National Cancer Institute, National Institutes of Health, under Contract No. HHSN261200800001E. The content of this code repository does not necessarily reflect the views or policies of the Department of Health and Human Services, nor does mention of trade names, commercial products, or organizations imply endorsement by the U.S. Government. 
