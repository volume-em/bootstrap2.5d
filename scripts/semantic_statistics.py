"""

Runs evaluation on the ground truth and inferred labels

"""

import torch
import os, shutil
import numpy as np
from tqdm import tqdm
from glob import glob
import SimpleITK as sitk
import pandas as pd
import argparse 
import matplotlib.pyplot as plt
import mlflow
from pandas import DataFrame

def import_file(fpath):
    reader = sitk.ImageFileReader()
    reader.SetFileName(fpath)
    imported = reader.Execute()
    return imported

def name_split(pred_name, gt_name, orig_name):
    pred_split = pred_name.split('/')
    gt_split = gt_name.split('/')
    orig_split = orig_name.split('/')
    return pred_split[-1], gt_split[-1], orig_split[-1]


def fn_clean(pred_fn, gt_fn):
    pred = pred_fn.split('.')
    gt = gt_fn.split('.')
    fn_cleaned = pred[0] + '_' + gt[0]
    return fn_cleaned

def copy_file_info(orig, pred):
    pred.CopyInformation(orig)
    return pred

def resample_file(gt, pred):
    pred_resamp = sitk.Resample(pred, gt)
    return pred_resamp

def image_to_array(gt, pred):
    gt_array = sitk.GetArrayFromImage(gt)
    pred_array = sitk.GetArrayFromImage(pred)
    if pred_array.max() > 1:
        pred_array = pred_array / 255.0     
    return gt_array, pred_array

def flatten_file(gt, pred):
    gt_flat = gt.ravel()
    pred_flat = pred.ravel()    
    return gt_flat, pred_flat

def dict_genarator():
    metrics = {}
    return metrics

def threshold_generator(metrics):
    metrics['Threshold'] = list(np.arange(0.1, 1, 0.1))
    return metrics

def test_statistics_generator(metrics):
    metrics['TP'] = []
    metrics['FP'] = []
    metrics['TN'] = [] 
    metrics['FN'] = []
    return metrics

def binary_classification(gt_flat, pred_flat_copy, metrics):
    TP = (gt_flat * pred_flat_copy).sum()
    FP = ((1 - gt_flat) * pred_flat_copy).sum()
    TN = ((1 - gt_flat) * (1 - pred_flat_copy)).sum()
    FN = (gt_flat * (1 - pred_flat_copy)).sum()
    metrics['TP'].append(TP)
    metrics['FP'].append(FP)
    metrics['TN'].append(TN)
    metrics['FN'].append(FN)
    return metrics

def calc_TPR(metrics):
    nume = np.asarray(metrics['TP'])
    denom = np.asarray(metrics['TP']) + np.asarray(metrics['FN']) + 1e-3
    metrics['TPR'] = list(nume / denom)
    return metrics

def calc_TNR(metrics):
    nume = np.asarray(metrics['TN'])
    denom = np.asarray(metrics['FP']) + np.asarray(metrics['TN']) + 1e-3
    metrics['TNR'] = list(nume / denom)
    return metrics

def calc_precision(metrics):
    nume = np.asarray(metrics['TP'])
    denom = np.asarray(metrics['FP']) + np.asarray(metrics['TP']) + 1e-3
    metrics['Precision'] = list(nume / denom)
    return metrics

def calc_recall(metrics):
    nume = np.asarray(metrics['TP'])
    denom = np.asarray(metrics['FN']) + np.asarray(metrics['TP']) + 1e-3
    metrics['Recall'] = list(nume / denom)
    return metrics

def calc_iou(metrics):
    inter = np.asarray(metrics['TP'])
    union = np.asarray(metrics['TP']) + np.asarray(metrics['FP']) + np.asarray(metrics['FN']) + 1e-3
    metrics['IoU'] = list(inter / union)
    return metrics

def calc_dice(metrics):
    nume = np.asarray(metrics['TP']) *2
    denom = (2 * np.asarray(metrics['TP'])) + np.asarray(metrics['FP']) + np.asarray(metrics['FN'])
    metrics['Dice Coefficient'] = list(nume / denom)
    return metrics 

def calc_bacc(metrics):
    left = np.asarray(metrics['TP']) / (np.asarray(metrics['TP']) + np.asarray(metrics['FN']))
    right = np.asarray(metrics['TN']) / (np.asarray(metrics['FP']) + np.asarray(metrics['TN']))
    metrics['Balanced Accuracy'] = list((left + right)/2)
    return metrics

def drop_statistics_column(metrics):
    df = DataFrame(metrics, columns = list(metrics.keys()))
    df = df.drop(['TP', 'FP', 'TN', 'FN'], axis = 1)
    return df
                   
def plot_iou_threshold(metrics, fn_cleaned):
    plt.figure()
    plt.plot(metrics['Threshold'], metrics['IoU'], c = '#0BF5D8', lw = 3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Threshold')
    plt.ylabel('IoU')
    plt.title('IoU Curve', fontsize = 20)
    plot_name = fn_cleaned + '_IoU_Curve.png'
    plt.savefig(plot_name)
    print('Intersection over Union Curve Plot saved as: ' + plot_name + '.')
    
def plot_prec_rec_threshold(metrics, fn_cleaned):
    plt.figure()
    plt.plot(metrics['Threshold'], metrics['TPR'], c = '#12E412', lw = 3, label = 'Recall')
    plt.plot(metrics['Threshold'], metrics['Precision'], c = '#0BF5D8', lw = 3, label = 'Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Threshold') 
    plt.ylabel('Score')
    plt.legend(loc="best", framealpha = 0.1, fontsize = 12)
    plt.title('Precision-Recall Curve', fontsize = 20)
    plot_name = fn_cleaned + '_Prec-Recall_Curve.png'
    plt.savefig(plot_name)
    print('Precision Recall Curve Plot saved as: ' + plot_name + '.')

def plot_dice_threshold(metrics, fn_cleaned):
    plt.figure()
    plt.plot(metrics['Threshold'], metrics['Dice Coefficient'], c = '#0BF5D8', lw = 3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Threshold')
    plt.ylabel('Dice Coefficient')
    plt.title('Dice Coefficient Curve', fontsize = 20)    
    plot_name = fn_cleaned + '_DC_Curve.png'
    plt.savefig(plot_name)
    print('Dice Coefficient Curve Plot saved as: ' + plot_name + '.')
    
def plot_bacc_threshold(metrics, fn_cleaned):
    plt.figure()
    plt.plot(metrics['Threshold'], metrics['Balanced Accuracy'], c = '#0BF5D8', lw = 3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Threshold')
    plt.ylabel('Balanced Accuracy')
    plt.title('Balanced Accuracy Curve', fontsize = 20) 
    plot_name = fn_cleaned + '_BA_Curve.png'
    plt.savefig(plot_name)
    print('Balanced Accuracy Curve Plot saved as: ' + plot_name + '.')
    
def tocsv(df, save_path):
    #semantic_stat_name =  fn_cleaned + '_semantic_stat.csv'
    export_csv = df.to_csv(save_path, index = None, header=True)
    return export_csv

def parse_args():
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Train a model on given dataset')
    parser.add_argument('impath', type=str, metavar='impath', help='Path to image file')
    parser.add_argument('pred_path', type=str, metavar='pred_path', help='Path to predicted segmentation')
    parser.add_argument('gt_path', type=str, metavar='gt_path', help='Path to gt segmentation')
    #parser.add_argument('dataset_name', type=str, metavar='dataset_name', help='Prefix to dataset name')
    parser.add_argument('save_path', type=str, metavar='save_path', help='Path to save stats')
    args = vars(parser.parse_args())
    
    return args

def snakemake_args():
    params = {}
    params['impath'] = snakemake.input[0]
    params['gt_path'] = snakemake.input[1]
    params['pred_path'] = snakemake.input[2]
    params['save_path'] = snakemake.output[0]
    
    return params

args = parse_args()
#args = snakemake_args()

orig_name = args['impath']
gt_name = args['gt_path']

### clean file names
pred_name = args['pred_path']
save_path = args['save_path']


pred_fn, gt_fn, orig_fn = name_split(pred_name, gt_name, orig_name)

###make a plots directory if there isn't one
#if not os.path.exists(plots_path):
#    os.mkdir(plots_path)

### import all files
orig = import_file(orig_name)
gt = import_file(gt_name) > 0
pred = import_file(pred_name)

### clean files before comparison
#pred = align_axes(orig, pred)
pred_copy = copy_file_info(orig, pred)
pred_resamp = resample_file(gt, pred_copy)
gt_array, pred_array = image_to_array(gt, pred_resamp)
gt_flat, pred_flat = flatten_file(gt_array, pred_array)

### prepare lists and dictionaries for each statistic
metrics = dict_genarator()
metrics = threshold_generator(metrics)
metrics = test_statistics_generator(metrics)

### apply threshold cutoff for label map
for i in tqdm(metrics['Threshold']):
    pred_flat_copy = pred_flat >= i
    metrics = binary_classification(gt_flat, pred_flat_copy, metrics)

### calculate statistics 
metrics = calc_precision(metrics)
metrics = calc_recall(metrics)
metrics = calc_TPR(metrics)
metrics = calc_TNR(metrics)
metrics = calc_iou(metrics)
metrics = calc_dice(metrics)
metrics = calc_bacc(metrics)

### plots for statistics per threshold
#fn_cleaned = os.path.join(plots_path, gt_fn.split('.')[0])
#plot_iou_threshold(metrics, fn_cleaned)
#plot_prec_rec_threshold(metrics, fn_cleaned)
#plot_dice_threshold(metrics, fn_cleaned)
#plot_bacc_threshold(metrics, fn_cleaned)


### export to csv file
df = drop_statistics_column(metrics)
tocsv(df, save_path)

#log all the outputs in mlflow
#mlflow.set_experiment(experiment)
#mlflow.start_run(run_id=run_id)
#mlflow.log_artifacts(plots_path)

#close the run
#mlflow.end_run()