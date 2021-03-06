#!/bin/bash

# Create directory variables
urocell_data_dir=$1;
    target_dir="${urocell_data_dir}/target/";
        target_img_dir="${urocell_data_dir}/target/images";
    train_dir="${urocell_data_dir}/train";
        train_img_dir="${train_dir}/images";

# Create lyso and mito directories
train_lyso_dir=$train_dir/lyso; mkdir -p $train_lyso_dir
train_mito_dir=$train_dir/mito; mkdir -p $train_mito_dir
target_lyso_dir=$target_dir/lyso; mkdir -p $target_lyso_dir
target_mito_dir=$target_dir/mito; mkdir -p $target_mito_dir

echo "Downloading UroCell dataset ..."

# Download test set
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/data/fib1-0-0-0.nii.gz -P $target_img_dir
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/lyso/fib1-0-0-0.nii.gz -P $target_lyso_dir
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/mito/fib1-0-0-0.nii.gz -P $target_mito_dir

#Download training sets
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/data/fib1-1-0-3.nii.gz -P $train_img_dir
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/lyso/fib1-1-0-3.nii.gz -P $train_lyso_dir
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/mito/fib1-1-0-3.nii.gz -P $train_mito_dir

wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/data/fib1-3-2-1.nii.gz -P $train_img_dir
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/lyso/fib1-3-2-1.nii.gz -P $train_lyso_dir
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/mito/fib1-3-2-1.nii.gz -P $train_mito_dir
    
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/data/fib1-3-3-0.nii.gz -P $train_img_dir
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/lyso/fib1-3-3-0.nii.gz -P $train_lyso_dir
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/mito/fib1-3-3-0.nii.gz -P $train_mito_dir

wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/data/fib1-4-3-0.nii.gz -P $train_img_dir
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/lyso/fib1-4-3-0.nii.gz -P $train_lyso_dir
wget -q -nc https://github.com/MancaZerovnikMekuc/UroCell/raw/master/mito/fib1-4-3-0.nii.gz -P $train_mito_dir