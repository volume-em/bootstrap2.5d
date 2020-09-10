import torch
import torch.nn as nn

class BootstrapDiceLoss(nn.Module):
    """
    Calculates the bootstrapped dice loss between model output logits and
    a noisy ground truth labelmap. The loss targets are modified to be
    a linear combination of the noisy ground truth the model's own prediction
    confidence. They are calculated as:
    
    boot_target = (beta * noisy_ground_truth) + (1.0 - beta) * model_predictions [1]
    
    References: 
    [1] https://arxiv.org/abs/1412.6596
    
    Arguments:
    ----------
    beta: Float, in the range (0, 1). Beta = 1 is equivalent to regular dice loss. 
    Controls the level of mixing between the noisy ground truth and the model's own 
    predictions. Default 0.8.
    
    eps: Float. A small float value used to prevent division by zero. Default, 1e-7.
    
    mode: Choice of ['hard', 'soft']. In the soft mode, model predictions are 
    probabilities in the range [0, 1]. In the hard mode, model predictions are 
    "hardened" such that:
    
        model_predictions = 1 when probability > 0.5
        model_predictions = 0 when probability <= 0.5
    
    Default is 'hard'.
    
    Example Usage:
    --------------
    
    model = Model()
    criterion = BootstrapDiceLoss(beta=0.8, mode='hard')
    output = model(input)
    loss = criterion(output, noisy_ground_truth)
    loss.backward()
    
    """
    def __init__(self, beta=0.8, eps=1e-7, mode='hard'):
        super(BootstrapDiceLoss, self).__init__()
        self.beta = beta
        self.eps = eps
        self.mode = mode
        
    def forward(self, output, target):
        #make target the same shape as output by unsqueezing
        #the channel dimension, if needed
        if target.ndim == output.ndim - 1:
            target = target.unsqueeze(1)
            
        assert(output.size() == target.size()), \
        "Output and target tensors must have the same size!"
            
        #get the number of classes from the output channels
        n_classes = output.shape[1]
        #1 output channel (i.e. binary classification) needs to be considered 
        #as 2 distinct labels
        n_classes = 2 if n_classes == 1 else n_classes
        
        #determine which dimensions are empty for creating the
        #one hot encoding
        empty_dims = (1,) * (target.ndim - 2)
        
        #one-hot encode the target (B, 1, H, W) --> (B, N, H, W)
        k = torch.arange(0, 2).view(1, 2, *empty_dims).to(target.device)
        target = (target == k)
            
        #convert output logits to probabilities using sigmoid for binary
        #classification and softmax for mutliclass
        if n_classes == 2:
            #get the positive and negative output probabilities with a sigmoid
            pos_prob = torch.sigmoid(output)
            neg_prob = 1 - pos_prob
            probas = torch.cat([neg_prob, pos_prob], dim=1)
        else:
            #for multiclass apply the softmax to get probabilities for each class
            probas = F.softmax(output, dim=1)
        
        #cast the 1-hot mask to the same type as the output
        target = target.type(output.dtype)
                
        #modify the target to its bootstrapped equivalent
        #when mode is soft, we take the weighted average
        #of the noisy ground truth and output probabilities [0, 1].
        #when mode is hard, we take the weighted average
        #of the noisy ground truth and the hardened output probabilities {0, 1}
        if self.mode == 'soft':
            boot_target = (self.beta * target) + (1.0 - self.beta) * probas
        else:
            boot_target = (self.beta * target) + (1.0 - self.beta) * (probas > 0.5).float()
        
        #sum over all dimensions aside from the channel dim
        dims = (0,) + tuple(range(2, boot_target.ndimension()))
        intersection = torch.sum(probas * boot_target, dims)
        cardinality = torch.sum(probas + boot_target, dims)
        
        #return the loss
        dice_loss = ((2. * intersection) / (cardinality + self.eps)).mean()
        return (1 - dice_loss)