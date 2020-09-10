# Bootstrap 2.5d
Code for the paper: [Enforcing Prediction Consistency Across Orthogonal Planes Significantly Improves Segmentation of FIB-SEM Image Volumes by 2D Neural Networks.](https://www.cambridge.org/core/journals/microscopy-and-microanalysis/article/enforcing-prediction-consistency-across-orthogonal-planes-significantly-improves-segmentation-of-fibsem-image-volumes-by-2d-neural-networks/97314A3AF09213E4E97491D95BF03C1B)

## Dependencies

By default, this code is designed to run with an NVidia GPU; it will not work on CPU-only machines without modification.





## Using this code

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

Next, move training image and labelmap image volumes into the data/train/images and data/train/masks directories, respectively. Note that the image and labelmap volume files must have exactly the same names and must have 8-bit unsigned voxels. Lastly, move a target image volume for segmentation into the data/target/images directory. Multiple volumes can be in each directory, training/inference will be applied to each as a group. See [here](https://simpleitk.readthedocs.io/en/master/IO.html) for a list of supported file formats.

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

Once the directories are setup, the last step is to run snakemake with the given Snakefile.
```
snakemake
```

The Snakefile has multiple outputs. The most important are the pytorch model state dicts and predicted segmentation volumes. The model state dicts will appear in the models/ directory. One will be called supervised.pth and the other called weakly_supervised.pth (these names should be obvious after reading the paper). The predicted segmentation volumes will appear in the target\ directory under super_preds/ and weaksuper_preds/.

To rerun the Snakefile with different data or hyperparameters, it may be necessary to remove all previously generated snakemake results. This can be done manually if only certain output need to be modified. For example, to repredict the weaksuper_preds/ only, just remove that directory:
```
rm -r data/target/weaksuper_preds/
```

To remove all outputs from the entire pipeline run:
```
snakemake --delete-all-output
rm -r data/valid2d/
```

The directory structure and pipeline hyperparameters can easily be modified by editing the Snakefile.