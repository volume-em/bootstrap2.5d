import torch
import torch.nn as nn
from matplotlib import pyplot as plt
#from .lovasz_losses import lovasz_softmax

class BootstrappedBCE(nn.Module):
    """
    Implements Bootstrapped BCE loss without the size weighting parameter
    
    """
    def __init__(self, kfactor=0.15):
        super(BootstrappedBCE, self).__init__()
        self.kfactor = kfactor
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, output, target):
        #flatten the output and target tensors
        bsz = output.size(0)
                
        output = output.view(bsz, -1)
        target = target.view(bsz, -1)
        
        #evaluate the BCE loss between output and target
        bce_loss = self.bce(output, target)
        
        #calculate the top k samples to evaluate
        k = int(self.kfactor * output.size(1))
        
        #calculate the top k values over each image
        topk = torch.topk(bce_loss, k, dim=1).values
        
        return topk.mean()
    
class BootstrappedCE(nn.Module):
    """
    Implements Bootstrapped CE loss without the size weighting parameter
    
    """
    def __init__(self, kfactor=0.15):
        super(BootstrappedCE, self).__init__()
        self.kfactor = kfactor
        self.ce = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, output, target):
        #flatten the output and target tensors
        bsz = output.size(0)
        n_classes = output.size(1)
        #output = output.view(bsz, num_classes, -1)
        #target = target.view(bsz, -1)
        
        if target.dtype != torch.long:
            target = target.long()
        
        #evaluate the CE loss between output and target
        ce_loss = self.ce(output, target) #(B, H, W)
        
        if target.ndim == output.ndim - 1:
            target = target.unsqueeze(1)
        
        #one hot encode to n_classes channels for multiclass
        empty_dims = (1,) * (target.ndim - 2)
        k = torch.arange(0, n_classes).view(1, n_classes, *empty_dims).to(target.device)
        target = (target == k) #(B, C, H, W)

        #apply the softmax to get probabilities
        ce_loss = ce_loss.unsqueeze(1) * target #(B, C, H, W)
        ce_loss = ce_loss.transpose(1, 0).contiguous().reshape(n_classes, -1)
        
        #calculate the top k samples to evaluate
        k = int(self.kfactor * ce_loss.size(1))
        
        #calculate the top k values over each image
        topk = torch.topk(ce_loss, k, dim=1).values
        
        return topk.mean()
    
class WeightedCE(nn.Module):
    """
    Implements Bootstrapped CE loss without the size weighting parameter
    
    """
    def __init__(self, weight=None):
        super(WeightedCE, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        
    def forward(self, output, target):
        #flatten the output and target tensors
        #get the weights from the target
        target = target.long()
        loss = self.ce(output, target)
        
        #target = target.unsqueeze(1).long()
        #empty_dims = (1,) * (target.ndim - 2)
        
        #k = torch.arange(0, output.size(1)).view(1, output.size(1), *empty_dims).to(target.device)
        #target = (target == k)
        
        #dims = (0,) + tuple(range(2, target.ndimension()))
        #print(loss.size(), target.size())
        #thing = loss.unsqueeze(1) * target.float()
        #print(thing.size())
        #print(torch.sum(thing, dims))
        
        return loss#.mean()
    
#class LovaszSoftmax(nn.Module):
#    """
#    Implements Lovasz softmax
#    
#    """
#    def __init__(self):
#        super(LovaszSoftmax, self).__init__()
#        self.softmax = nn.Softmax(dim=1)
#        
#    def forward(self, output, target):
#        #flatten the output and target tensors
#        proba = self.softmax(output)
#        
#        return lovasz_softmax(proba, target)
#
class DiceLoss(nn.Module):
    """
    Computes the Sørensen–Dice loss.
    
    Note that PyTorch optimizers minimize a loss. In this case, we would 
    like to maximize the dice loss so we return the negated dice loss.
    Args:
    -----
    
    target: a tensor of shape [B, 1, H, W]. One hot encoded if multiple labels.
    output: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logits of the model.
    eps: added to the denominator for numerical stability.
    
    Returns:
    --------
    
    dice_loss: the Sørensen–Dice loss.
    
    """
    def __init__(self, eps=1e-5, classes='present'):
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.classes = classes
        
    def forward(self, output, target):
        #make target the same shape as output by unsqueezing
        #the channel dimension, if needed
        if target.ndim == output.ndim - 1:
            target = target.unsqueeze(1)
        
        #get the number of classes from the output channels
        n_classes = output.shape[1]
        
        #get reshape size based on number of dimensions
        #can exclude first 2 dims, which are also batch and channel
        empty_dims = (1,) * (target.ndim - 2)
        
        if n_classes == 1:
            #one-hot encode the target (B, 1, H, W) --> (B, N, H, W)
            k = torch.arange(0, 2).view(1, 2, *empty_dims).to(target.device)
            target = (target == k)
            
            #get the output probabilities with a sigmoid
            pos_prob = torch.sigmoid(output)
            neg_prob = 1 - pos_prob
            probas = torch.cat([neg_prob, pos_prob], dim=1)
        else:
            #one hot encode to n_classes channels for multiclass
            k = torch.arange(0, n_classes).view(1, n_classes, *empty_dims).to(target.device)
            target = (target == k)
            
            #apply the softmax to get probabilities
            probas = nn.Softmax(dim=1)(output)
        
        #cast the 1-hot mask to the right type
        target = target.type(output.dtype)
        
        #operate over all dimensions aside from the channel dim
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(probas * target, dims) #(C)
        cardinality = torch.sum(probas + target, dims) #(C)
        
        if self.classes == 'present':
            indices = torch.where(torch.sum(target, dims) > 10)[0]
            intersection = intersection[indices]
            cardinality = cardinality[indices]
            weight = torch.sum(target, dims)[indices] ** -2
            #print(weight)
            dice_loss =  1 - ((2. * weight * intersection + self.eps) / (weight * cardinality + self.eps))
        else:
            #compute the negated dice loss
            dice_loss =  1 - ((2. * intersection + self.eps) / (cardinality + self.eps))
            
        #print(dice_loss)
        
        return dice_loss.mean()
    
class CEDiceLoss(nn.Module):
    def __init__(self, weight=None, eps=1e-5):
        super(CEDiceLoss, self).__init__()
        self.dice = DiceLoss(eps)
        self.ce = nn.CrossEntropyLoss(weight=weight)
        
    def forward(self, output, target):
        dice = self.dice(output, target)
        ce = self.ce(output, target.long())
        
        return 0.7 * dice + 0.3 * ce
    
class SSLoss(nn.Module):
    def __init__(self, logits=True):
        super(SSLoss, self).__init__()
        self.logits = logits
        
    def forward(self, output, target):
        bsz = output.size(0)
        #flatten tensors to start
        output = output.view(bsz, -1)
        target = target.view(bsz, -1)
        
        #apply logits, if needed
        if self.logits:
            output = nn.Sigmoid()(output)
            
        #compute the l2 norm between output and
        #target and then square it and average
        #(B, -1) --> (B, )
        loss = ((target - output) ** 2).mean()
        
        return loss
    
class BootstrapDiceLoss(nn.Module):
    """
    Calculates the dice loss between model predictions and target. Technically,
    because the loss function uses prediction probabilities in range [0, 1], it 
    is a soft dice loss. 
    
    NOTE: Applies Sigmoid activation to the model predictions.
    
    Arguments:
    eps: A smoothing parameter added to numerator and denominator in order to
    prevent NaN values (i.e. division by zero). Default value is 1.
    
    Forward pass expects model output and target. Output and target tensors must have the same
    shape.
    
    Returns:
    Dice loss float value for the entire input batch divided by the batch size.
    
    Example:
    
    dice = DiceLoss(eps=1.)
    loss = dice(output, target)
    loss.backward()
    
    """
    def __init__(self, beta=0.8, eps=1e-7):
        super(BootstrapDiceLoss, self).__init__()
        self.beta = beta
        self.eps = eps
        self.dice_loss = DiceLoss(eps=eps)
        
    def forward(self, output, target):
        #flatten both tensors
        #bsz = output.size(0)
        #the only difference between regular
        #and bootstrap dice loss is that bootstrap
        #dice loss has a label "softening" parameter
        #called beta
        #we tweak the target variable to account
        #for the possibility of noise, ideally, beta
        #should equal the value of IoU in 3d
        target = self.beta * target + (1 - self.beta) * nn.Sigmoid()(output)
        
        return self.dice_loss(output, target)
    
class HardDiceLoss(nn.Module):
    """
    Loss(t, p) = - (beta * t + (1 - beta) * z) * log(p)
    where z = argmax(p)
    """
    def __init__(self, beta=0.8, eps=1e-7):
        super(HardDiceLoss, self).__init__()
        self.beta = beta
        self.dice_loss = DiceLoss(eps=eps)

    def forward(self, output, target):
        # cross_entropy = - t * log(p)
        bsz = output.size(0)
        beta_dice = self.beta * self.dice_loss(output, target)

        # z = argmax(p)
        sigmoid_output = nn.Sigmoid()(output.view(bsz, -1))
        _, z = torch.max(sigmoid_output, dim=1)
        z = z.view(-1, 1)
        bootstrap = sigmoid_output.gather(1, z).view(-1)
        # second term = (1 - beta) * z * log(p)
        bootstrap = - (1.0 - self.beta) * bootstrap

        return torch.mean(beta_dice + bootstrap)
    
class HardDiceLoss2(nn.Module):
    def __init__(self, beta=0.8, eps=1e-5, threshold=0.5):
        super(HardDiceLoss2, self).__init__()
        self.beta = beta
        self.eps = eps
        self.thr = threshold
        
    def forward(self, output, target):
        #make target the same shape as output by unsqueezing
        #the channel dimension, if needed
        if target.ndim == output.ndim - 1:
            target = target.unsqueeze(1)
            
        #get the number of classes from the output channels
        n_classes = output.shape[1]
        
        #get reshape size based on number of dimensions
        #can exclude first 2 dims, which are also batch and channel
        empty_dims = (1,) * (target.ndim - 2)
        
        if n_classes == 1:
            #one-hot encode the target (B, 1, H, W) --> (B, N, H, W)
            k = torch.arange(0, 2).view(1, 2, *empty_dims).to(target.device)
            target = (target == k)
            
            #get the output probabilities with a sigmoid
            pos_prob = torch.sigmoid(output)
            neg_prob = 1 - pos_prob
            probas = torch.cat([neg_prob, pos_prob], dim=1)
        else:
            #one hot encode to n_classes channels for multiclass
            k = torch.arange(0, n_classes).view(1, n_classes, *empty_dims).to(target.device)
            target = (target == k)
            
            #apply the softmax to get probabilities
            probas = F.softmax(output, dim=1)
        
        #cast the 1-hot mask to the right type
        target = target.type(output.dtype)
                
        #to convert to the hard case, we
        #threshold the output at 0.5
        boot_target = self.beta * target + (1.0 - self.beta) * (probas > self.thr).float()
        
        #operate over all dimensions aside from the channel dim
        dims = (0,) + tuple(range(2, boot_target.ndimension()))
        intersection = torch.sum(probas * boot_target, dims)
        cardinality = torch.sum(probas + boot_target, dims)
        
        #compute the negated dice loss
        dice_loss = ((2. * intersection) / (cardinality + self.eps)).mean()
        return (1 - dice_loss)

class SemiDiceLoss(nn.Module):
    """
    Computes the Sørensen–Dice loss on soft target probabilities (including pseudo-labels)
    
    Note that PyTorch optimizers minimize a loss. In this case, we would 
    like to maximize the dice loss so we return the negated dice loss.
    Args:
    -----
    
    target: a tensor of shape [B, 1, H, W]. One hot encoded if multiple labels.
    output: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logits of the model.
    eps: added to the denominator for numerical stability.
    
    Returns:
    --------
    
    dice_loss: the Sørensen–Dice loss.
    
    """
    def __init__(self, eps=1e-7):
        super(SemiDiceLoss, self).__init__()
        self.eps = eps
        
    def forward(self, output, target):
        #make target the same shape as output by unsqueezing
        #the channel dimension, if needed
        #if target.ndim == output.ndim - 1:
        #    target = target.unsqueeze(1)
        
        #get the number of classes from the output channels
        n_classes = output.shape[1]
        
        #get reshape size based on number of dimensions
        #can exclude first 2 dims, which are also batch and channel
        empty_dims = (1,) * (target.ndim - 2)
        
        if n_classes == 1:
            #one-hot encode the target (B, 1, H, W) --> (B, 2, H, W)
            #k = torch.arange(0, 2).view(1, 2, *empty_dims).to(target.device)
            #target = (target == k)
            
            #get the output probabilities with a sigmoid
            pos_prob = torch.sigmoid(output)
            neg_prob = 1 - pos_prob
            probas = torch.cat([neg_prob, pos_prob], dim=1)
        else:
            #one hot encode to n_classes channels for multiclass
            #k = torch.arange(0, n_classes).view(1, n_classes, *empty_dims).to(target.device)
            #target = (target == k)
            
            #apply the softmax to get probabilities
            probas = F.softmax(output, dim=1)
        
        #cast the 1-hot mask and pseudo mask to the right type
        target = target.type(output.dtype)
        
        #if pseudo_target was given, cast it and concat to target
        #the pseudo target will already have the correct shape
        #assumes pseudo images appended to the end of output
        #if pseudo_target is not None:
        #    pseudo_target = pseudo_target.type(output.dtype)
        #    target = torch.cat([target, pseudo_target], dim=0)
        
        #operate over all dimensions aside from the channel dim
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(probas * target, dims)
        cardinality = torch.sum(probas + target, dims)
        
        #compute the negated dice loss
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return (1 - dice_loss)
    
def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: str = None,
            dtype: torch.dtype = None,
            eps: float = 1e-6) -> torch.Tensor:
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.

    Returns:
        torch.Tensor: the labels in one hot tensor of shape :math:`(N, C, *)`,

    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> kornia.losses.one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    shape = labels.shape
    one_hot = torch.zeros(shape[0], num_classes, *shape[1:],
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
    
def focal_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: float,
        gamma: float = 2.0,
        reduction: str = 'none',
        eps: float = 1e-8) -> torch.Tensor:
    r"""Function that computes Focal loss.

    See :class:`~kornia.losses.FocalLoss` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                         .format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(
            out_size, target.size()))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}" .format(
                input.device, target.device))

    # compute softmax over the classes axis
    input_soft: torch.Tensor = nn.Softmax(dim=1)(input) + eps

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(
        target, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1., gamma)
    #alpha = alpha[None, :, None, None]

    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}"
                                  .format(reduction))
    return loss


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to [1], the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.


    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> loss = kornia.losses.FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float, gamma: float = 2.0,
                 reduction: str = 'none') -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = 1e-6

    def forward(  # type: ignore
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        target = target.long()
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)
    
#class JointLoss(nn.Module):
#    def __init__(self):
#        super(JointLoss, self).__init__()
#        self.focal = nn.CrossEntropyLoss()
#        self.lovasz = LovaszSoftmax()
#
#    def forward(self, output, target):
#        f = self.focal(output, target)
#        l = self.lovasz(output, target)
#        print(f, l)
#
#        return 0.5 * f + 0.5 * l
