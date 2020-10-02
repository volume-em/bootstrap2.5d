import torch
import torch.nn as nn

class AverageMeter:
    """Computes average values"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum = self.sum + self.val
        self.count += 1
        self.avg = self.sum / self.count

class EMAMeter:
    """Computes and stores an exponential moving average and current value"""
    def __init__(self, momentum=0.98):
        self.mom = momentum
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum = (self.sum * self.mom) + (val * (1 - self.mom))
        self.count += 1
        self.avg = self.sum / (1 - self.mom ** (self.count))

def calculate_iou(output, target):
    """
    Calculates the mean iou of all classes in the segmentation mask.
    
    Arguments:
    ----------
    output: Logits of predicted segmentation mask of size (B, N_CLASSES, H, W).
    target: Ground truth segmentation mask of size (B, H, W) or (B, 1, H, W).
    
    Returns:
    --------
    mean_iou: Float in range [0, 1].
    
    """
    #make target the same shape as output by unsqueezing
    #the channel dimension, if needed
    if target.ndim == output.ndim - 1:
        target = target.unsqueeze(1)

    #get the number of classes from the output channels
    n_classes = output.size(1)

    #get reshape size based on number of dimensions
    #can exclude first 2 dims, which are always batch and channel
    empty_dims = (1,) * (target.ndim - 2)

    if n_classes > 1:
        #softmax the output and argmax
        output = nn.Softmax(dim=1)(output) #(B, NC, H, W)
        max_idx = torch.argmax(output, 1, keepdim=True) #(B, 1, H, W)

        #one hot encoder the target and output
        target = target.long()
        target_onehot = torch.zeros_like(target)
        target = target_onehot.scatter(1, target, 1)
        output_onehot = torch.zeros_like(output)
        output = output_onehot.scatter(1, max_idx, 1)
    else:
        #just sigmoid the output
        output = (nn.Sigmoid()(output) > 0.5).long()

    #cast target to the correct type for operations
    target = target.type(output.dtype)

    #multiply the tensors, everything that is still as 1 is part of the intersection
    #(N,)
    dims = (0,) + tuple(range(2, target.ndimension()))
    intersect = torch.sum(output * target, dims)

    #compute the union, (N,)
    union = torch.sum(output + target, dims) - intersect

    #avoid division errors by adding a small epsilon
    #also guarantees that iou = 1 when ground truth
    #and prediction segmentations are empty
    iou = (intersect + 1e-5) / (union + 1e-5)

    return iou.mean().item()