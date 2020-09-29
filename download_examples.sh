#!/bin/bash

# Create models and data directories
mkdir -p models;
mkdir -p data;
mkdir -p data/train;
    mkdir -p data/train/images;
    mkdir -p data/train/masks;
mkdir -p data/target;
    mkdir -p data/target/images;

# Create directory variables
urocell_data_dir="data";
    target_dir="${urocell_data_dir}/target/images";
    train_dir="${urocell_data_dir}/train";
        train_img_dir="${train_dir}/images";
        train_msk_dir="${train_dir}/masks";

# Create lyto and mito directories
lyso_dir=$train_msk_dir/lyso; mkdir -p $lyso_dir
mito_dir=$train_msk_dir/mito; mkdir -p $mito_dir

echo "Downloading UroCell dataset ..."

# Download test set
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/data/fib1-0-0-0.nii.gz -P $target_dir

#Download training sets
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/data/fib1-1-0-3.nii.gz -P $train_img_dir
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/lyso/fib1-1-0-3.nii.gz -P $lyso_dir
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/mito/fib1-1-0-3.nii.gz -P $mito_dir

wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/data/fib1-3-2-1.nii.gz -P $train_img_dir
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/lyso/fib1-3-2-1.nii.gz -P $lyso_dir
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/mito/fib1-3-2-1.nii.gz -P $mito_dir
    
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/data/fib1-3-3-0.nii.gz -P $train_img_dir
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/lyso/fib1-3-3-0.nii.gz -P $lyso_dir
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/mito/fib1-3-3-0.nii.gz -P $mito_dir

wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/data/fib1-4-3-0.nii.gz -P $train_img_dir
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/lyso/fib1-4-3-0.nii.gz -P $lyso_dir
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/mito/fib1-4-3-0.nii.gz -P $mito_dir