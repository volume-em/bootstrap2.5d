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
        #one-hot encode the target (B, 1, H, W) --> (B, N, H, W)
        k = torch.arange(0, n_classes).view(1, n_classes, *empty_dims).to(target.device)
        target = (target == k)

        #softmax the output
        output = nn.Softmax(dim=1)(output)
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
    iou = (intersect + 1e-7) / (union + 1e-7)

    return iou