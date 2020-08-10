import torch
import torch.nn as nn
from copy import deepcopy
from matplotlib import pyplot as plt

import numpy as np
import skimage.measure as measure

class Metric:
    """
    This is the base class for metrics. A Metric subclass should have an __init__ function
    and a calculate function that computes an integer/float value and calls self.batch_end
    to store the metric for later. See metric subclasses for examples.
    """
    def __init__(self):
        self.history = []
        #a running batch average of the metric, stored at the end of each epoch
        self.batch_avg = 0
        self.n = 0
        
    def epoch_end(self):
        #store metric and reinitialize for next epoch
        self.history.append(self.batch_avg)
        self.batch_avg = 0
        self.n = 0
        
    def batch_end(self, batch_metric):
        self.batch_avg = (self.batch_avg * self.n + batch_metric) / (self.n + 1)
        self.n += 1
        
    def clear_history(self):
        self.history = []
        
class IoU(Metric):
    """
    Computes IoU metric for batch of predictions and masks.
    
    Arguments:
    logits: If logits is True, then a sigmoid activation is applied to output
    before additional calculations. If False, then it is assumed that sigmoid
    activation has already been applied previously.
    
    Return:
    Calculate method adds calculated IoU value to a running average for epoch end
    evaluation.
    
    Usage:
    Case 1: Training loop
    
    iou = IoU()
    #NOTE: output.size == target.size
    for batch in epoch:
        ...
        iou.calculate(output, target)
        ...
    iou.epoch_end()
    print(iou.history) #prints the average value of IoU for all batches in the epoch
    
    Case 2: Single image or batch
    
    iou = IoU()
    #NOTE: output.size == target.size
    iou.calculate(output, target)
    iou.epoch_end()
    print(iou.history) #prints the value of IoU for output and target
    
    """
    
    def __init__(self, logits=True, is_guay=False, is_perez=False, masking=False):
        super(IoU, self).__init__()
        #if logits is True we apply a sigmoid to output to get probabilities
        self.logits = logits
        self.is_guay = is_guay
        self.is_perez = is_perez
        self.masking = masking
        
    def calculate(self, output, target):
        #get the number of classes from the output channels
        n_classes = output.size(1)
        
        #make sure the target has a channel dimension
        if target.ndim == (output.ndim - 1):
            target = target.unsqueeze(1)
                        
        #get reshape size based on number of dimensions
        #can exclude first 2 dims, which are always batch and channel
        empty_dims = (1,) * (target.ndim - 2)
            
        if n_classes > 1:
            #softmax the output
            output = nn.Softmax(dim=1)(output)
            
            #argmax
            max_idx = torch.argmax(output, 1, keepdim=True)
        else:
            #just sigmoid the output
            output = (nn.Sigmoid()(output) > 0.5).long()
        
        if self.is_perez and self.masking:
            labeled_mask = measure.label((target > 0).squeeze().cpu().numpy())
            labeled_pred = measure.label((output == 1).squeeze().cpu().numpy())

            total_intersect = 0
            total_union = 0
            for l in np.unique(labeled_mask)[1:]:
                lmask = labeled_mask == l
                plabels = np.unique(labeled_pred[lmask])[1:]
                lpred = np.zeros_like(labeled_pred)
                for pl in plabels:
                    lpred += labeled_pred == pl

                total_intersect += np.sum(lmask * lpred)
                total_union += np.sum(lmask) + np.sum(lpred)
                
            iou = (total_intersect + 1e-7) / (total_union - total_intersect + 1e-7)
            self.batch_end(np.array([iou]))
            return
        
        if self.is_guay and self.masking:
            ious = []
            max_idx[target == 0] = 0
            for label in [2, 3, 4, 5, 6]:
                label_pred = max_idx == label
                label_gt = target == label
                intersect = (label_pred * label_gt).sum()
                union = label_pred.sum() + label_gt.sum() - intersect
                iou = (intersect + 1e-7) / (union + 1e-7)
                ious.append(iou)
            
            self.batch_end(torch.Tensor(ious).float())
            return
            
        #one-hot encode (B, 1, H, W) --> (B, N, H, W)
        if n_classes > 1:
            k = torch.arange(0, n_classes).view(1, n_classes, *empty_dims).to(target.device)
            target = (target == k)
            
            output = torch.zeros_like(output)
            output = output.scatter(1, max_idx, 1)
            
        target = target.type(output.dtype)
        
        #multiply the tensors, everything that is still as 1 is part of the intersection
        #(N,)
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersect = torch.sum(output * target, dims)
        
        #compute the union, (N,)
        union = torch.sum(output + target, dims) - intersect
        
        #avoid division errors by adding a small epsilon
        iou = (intersect + 1e-7) / (union + 1e-7)
        #print(intersect, union, iou)
        
        #return the N IoUs for each class
        if self.is_guay:
            iou = iou[2:]
        
        self.batch_end(iou)
            
class Dice(Metric):
    """
    Computes Dice coefficient metric for batch of predictions and masks.
    
    Arguments:
    logits: If logits is True, then a sigmoid activation is applied to output
    before additional calculations. If False, then it is assumed that sigmoid
    activation has already been applied previously.
    
    Return:
    Calculate method adds calculated Dice coefficient value to a running average for 
    epoch end evaluation.
    
    Usage:
    Case 1: Training loop
    
    dice = Dice()
    #NOTE: output.size == target.size
    for batch in epoch:
        ...
        dice.calculate(output, target)
        ...
    dice.epoch_end()
    print(dice.history) #prints the average Dice coefficient for all batches in the epoch
    
    Case 2: Single image or batch
    
    dice = Dice()
    #NOTE: output.size == target.size
    dice.calculate(output, target)
    dice.epoch_end()
    print(dice.history) #prints the Dice coefficient for output and target
    
    """
    
    def __init__(self, logits=True):
        super(Dice, self).__init__()
        self.logits = logits
        
    def calculate(self, output, target):
        #get the number of classes from the output channels
        n_classes = output.size(1)
        
        #get reshape size based on number of dimensions
        #can exclude first 2 dims, which are also batch and channel
        empty_dims = (1,) * (target.ndim - 2)
        
        if n_classes > 1:
            #one-hot encode the target (B, 1, H, W) --> (B, N, H, W)
            k = torch.arange(0, n_classes).view(1, n_classes, *empty_dims).to(target.device)
            target = (target == k)
            
            #softmax the output
            output = nn.Softmax(dim=1)(output)
            
            #argmax and one-hot encode
            max_idx = torch.argmax(output, 1, keepdim=True)
            output = torch.zeros_like(output)
            output = output.scatter(1, max_idx, 1)
        else:
            #just sigmoid the output
            output = (nn.Sigmoid()(output) > 0.5).long()
            
        #cast target to the correct type for operations
        target = target.type(output.dtype)
        
        #now we need to calculate TP, FP, FN: (N, )
        dims = (0,) + tuple(range(2, target.ndimension()))
        TP = torch.sum(output * target, dims)
        FP = torch.sum(output * (1 - target), dims)
        FN = torch.sum(target * (1 - output), dims)
        
        #plus 1e-7 should only make a small change in value, but prevents nans
        dice = (2 * TP + 1e-7) / (2 * TP + FP + FN + 1e-7)
        
        self.batch_end(dice)
            
class BalancedAccuracy(Metric):
    """
    Computes Balanced Accuracy metric for batch of predictions and masks.
    
    Arguments:
    logits: If logits is True, then a sigmoid activation is applied to output
    before additional calculations. If False, then it is assumed that sigmoid
    activation has already been applied previously.
    
    Return:
    Calculate method adds calculated balanced accuracy to a running average for 
    epoch end evaluation.
    
    Usage:
    Case 1: Training loop
    
    ba = BalancedAccuracy()
    #NOTE: output.size == target.size
    for batch in epoch:
        ...
        ba.calculate(output, target)
        ...
    ba.epoch_end()
    print(ba.history) #prints the average balanced accuracy of all batches in the epoch
    
    Case 2: Single image or batch
    
    ba = BalancedAccuracy()
    #NOTE: output.size == target.size
    ba.calculate(output, target)
    ba.epoch_end()
    print(ba.history) #prints the balanced accuracy for output and target
    """
    
    def __init__(self, logits=True):
        super(BalancedAccuracy, self).__init__()
        self.logits = logits
        
    def calculate(self, output, target):
        #get the number of classes from the output channels
        n_classes = output.size(1)
        
        #get reshape size based on number of dimensions
        #can exclude first 2 dims, which are also batch and channel
        empty_dims = (1,) * (target.ndim - 2)
        
        if n_classes > 1:
            #one-hot encode the target (B, 1, H, W) --> (B, N, H, W)
            k = torch.arange(0, n_classes).view(1, n_classes, *empty_dims).to(target.device)
            target = (target == k)
            
            #softmax the output
            output = nn.Softmax(dim=1)(output)
            
            #argmax and one-hot encode
            max_idx = torch.argmax(output, 1, keepdim=True)
            output = torch.zeros_like(output)
            output = output.scatter(1, max_idx, 1)
        else:
            #just sigmoid the output
            output = (nn.Sigmoid()(output) > 0.5).long()
            
        #cast target to the correct type for operations
        target = target.type(output.dtype)
        
        #now we need to calculate TP, FP, FN, TN
        dims = (0,) + tuple(range(2, target.ndimension()))
        TP = torch.sum(output * target, dims)
        FP = torch.sum(output * (1 - target), dims)
        FN = torch.sum(target * (1 - output), dims)
        TN = torch.sum((1 - output) * (1 - target), dims)
        
        #plus 1 should only make a small change in value, but prevents nan values
        sens = (TP + 1e-7) / (TP + FN + 1e-7)
        spec = (TN + 1e-7) / (TN + FP + 1e-7)
        bacc = (sens + spec) / 2
        
        self.batch_end(bacc)
        
class ComposeMetrics:
    """
    A class for composing metrics together.
    
    Arguments:
    metrics_dict: A dictionary in which each key is the name of a metric and
    each value is a subclass of Metric with a calculate method that accepts
    model predictions and ground truth labels.
    
    Example:
    metrics_dict = {'IoU': IoU(), 'Dice coeff': Dice()}
    metrics = Compose(metrics_dict)
    #output.size == target.size
    metrics.evaluate(output, target)
    
    #to print value of IoU only
    #if close_epoch() is not called, history will be empty
    metrics.close_epoch()
    print(metrics.metrics['IoU'].history)
    
    #to print value of all metrics in metric_dict
    #if close_epoch() is not called, history will be empty
    metrics.close_epoch()
    print(metrics.print())
    """
    def __init__(self, metrics_dict, class_names=None, is_guay=False):
        self.metrics_dict = metrics_dict
        self.class_names = class_names
        
        if is_guay:
            self.class_names = class_names[2:]
        
    def evaluate(self, output, target):
        #make target the same shape as output by unsqueezing
        #the channel dimension, if needed
        if target.ndim == output.ndim - 1:
            target = target.unsqueeze(1)
        
        #calculate all the metrics in the dict
        for metric in self.metrics_dict.values():
            metric.calculate(output, target)
            
    def close_epoch(self):
        for metric in self.metrics_dict.values():
            metric.epoch_end()
            
            
    def print(self):
        names = []
        numbers = []
        for name, metric in self.metrics_dict.items():
            #we expect metric to be a tensor of size (n_classes,)
            #we want to print the corresponding class names if given
            if self.class_names is not None:
                for ix, class_name in enumerate(self.class_names):
                    names.append(class_name + ' ' + name + ' {} ')
                    numbers.append(metric.history[-1][ix])
            else:
                for ix in range(len(metric.history[-1])):
                    names.append('Class %d' % ix + ' ' + name + ' {} ')
                    numbers.append(metric.history[-1][ix])
                    
            if len(metric.history[-1]) > 1:
                print(f'Mean {name}: {metric.history[-1].mean()}')
                    
        s = ''.join(names)
        print(s.format(*numbers))
        print('\n')