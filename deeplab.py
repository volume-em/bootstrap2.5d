import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50

RESNET_MODELS = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50}

class conv1_relu_bn(nn.Module):
    """
    Pytorch layer that applies a 2d convolution with kernel size 1x1, stride 1
    and no padding. After the conv layer it applies ReLU activation and batchnorm.
    
    Arguments:
    -----------
    
    nin: Number of channels in input, used for Conv2d
    nout: Desired number of channels in output, used for Conv2d
    """
    def __init__(self, nin, nout):
        super(conv1_relu_bn, self).__init__()
        self.conv = nn.Conv2d(nin, nout, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(nout)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.bn(self.relu(self.conv(x)))
    
class conv3_relu_bn(nn.Module):
    """
    Pytorch layer that applies a 2d convolution with kernel size 3x3, stride 1
    and padding 1. After the conv layer it applies ReLU activation and batchnorm.
    
    Arguments:
    -----------
    
    nin: Number of channels in input, used for Conv2d
    nout: Desired number of channels in output, used for Conv2d
    """
    def __init__(self, nin, nout, padding=1, dilation=1):
        super(conv3_relu_bn, self).__init__()
        self.conv = nn.Conv2d(nin, nout, 3, 1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(nout)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.bn(self.relu(self.conv(x)))
    
class Encoder(nn.Module):
    def __init__(self, resnet_model='resnet34', pretrained=True):
        """
        Creates the ResNet encoder for DeepLabV3 model.
        
        Arguments:
        ----------
        resnet_model: String, choice of [resnet18, resnet34, resnet50]. Default is resnet34.
        
        pretrained: Bool. If True, resnet model is initialized with ImageNet pretrained
        weights. If False, resnet model is randomly initialized. Default is True.
        
        """
        super(Encoder, self).__init__()
        #change the strides in resnet such that the output of the encoder
        #is 16x smaller than the input (default in ResNet is 32x smaller)
        resnet = RESNET_MODELS[resnet_model]
        resnet = resnet(pretrained=pretrained)
        
        #there are two possibilities, either conv1 or conv2 will have a stride of 2, 
        #(e.g. for resnet 18,34 and resnet50 respectively)
        if resnet_model == 'resnet50':
            resnet.layer4[0].conv2.stride = (1, 1)
        else:
            resnet.layer4[0].conv1.stride = (1, 1)

        #the downsample layer always has stride (2, 2)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        
        #delete the average pooling and fc layers
        del resnet.avgpool
        del resnet.fc
        
        self.resnet = resnet
        
        #create layer groups
        #assuming input image is size 224x224, 
        #although any size divisible by 16 is OK
        self.groups = [
            nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu), #112, 112
            nn.Sequential(self.resnet.maxpool, self.resnet.layer1), #56, 56
            self.resnet.layer2, #28, 28
            self.resnet.layer3, #14, 14
            self.resnet.layer4 #14, 14
        ]
        
    def forward(self, x):
        #pass the image x, (B, 3, H, W)
        #through each encoder group and return
        #the feature maps
        features = []
        for group in self.groups:
            x = group(x)
            features.append(x)
        
        return features
    
class ImagePooling(nn.Module):
    """
    DeepLabV3 image pooling layer. Takes the output from the ResNet encoder and average pools
    it down to a 1x1 feature map.
    
    Arguments:
    ----------
    nin: Int, number of channels in the input.
    nout: Int, number of channels in the output
    pool_size: Tuple of Int. Size of the average pooled feature map. Default is (1, 1).
    
    """
    def __init__(self, nin, nout, pool_size=(1, 1)):
        super(ImagePooling, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.conv = conv1_relu_bn(nin, nout)
        
    def forward(self, x):
        #get height and width
        unpool_size = (x.size(2), x.size(3))
        #pool image from (B, C, H/16, W/16) --> (B, C, 1, 1)
        x = self.pool(x)
        x = self.conv(x)
        
        #return interpolated pooled image: (B, C, 1, 1) --> (B, C, H/16, W/16)
        return F.interpolate(x, size=unpool_size, mode='bilinear', align_corners=True)
    
class ASPP(nn.Module):
    """
    DeepLabV3 atrous spatial pyramid pooling layer.
    
    Arguments:
    ----------
    nin: Int, number of channels in the input.
    nout: Int, number of output channels from the module.
    drop_p: Float in [0, 1). Dropout probability to apply to the output of the ASPP module.
    dilations: List of Ints of length 3. Dilations factors of the convolutions in each
    module of the spatial pyramid. Default is [2, 4, 6]. Larger images with larger objects of
    interest may benefit from larger values of dilation. We're assuming images of size 224x224.
    
    """
    def __init__(self, nin, nout, drop_p=0, dilations=[2, 4, 6]):
        super(ASPP, self).__init__()
        self.conv1 = conv1_relu_bn(nin, nout)
        
        d1, d2, d3 = dilations
        self.atrous1 = conv3_relu_bn(nin, nout, padding=d1, dilation=d1)
        self.atrous2 = conv3_relu_bn(nin, nout, padding=d2, dilation=d2)
        self.atrous3 = conv3_relu_bn(nin, nout, padding=d3, dilation=d3)
        self.im_pool = ImagePooling(nin, nout, pool_size=1)
        
        self.groups = [
            self.conv1,
            self.atrous1,
            self.atrous2,
            self.atrous3,
            self.im_pool
        ]
        
        self.reduce = conv3_relu_bn(nout * 5, nout, 1, 1)
        self.drop = nn.Dropout2d(drop_p)
        
    def forward(self, x):
        #return the output features from each
        #module in the ASPP
        #x.size(): (B, NIN, H/16, W/16)
        features = []
        for group in self.groups:
            features.append(group(x))
        
        #size of each feature map must be (B, NOUT, H/16, W/16)
        #concatenate them along the channel dimension to get
        #(B, NOUT * 5, H/16, W/16)
        x = torch.cat(features, dim=1)
        #(B, NOUT * 5, H/16, W/16) --> (B, NOUT, H/16, W/16)
        return self.drop(self.reduce(x))
    
class Decoder(nn.Module):
    """
    Creates the Decoder for the DeepLabV3 model.

    Arguments:
    ----------
    aspp_nin: Int. Number of channels in the output of the ASPP module
    low_level_nin: Int. Number of channels in the low level feature map, which is
    the output of layer1 in the Encoder's resnet model.
    low_level_nout: Int. Number of channels in the skip connection that the low
    level feature map passes through.
    num_classes: Int. Number of output classes for the segmentation model.

    """
    def __init__(self, aspp_nin=64, low_level_nin=64, low_level_nout=32, num_classes=1):
        super(Decoder, self).__init__()
        self.skip = conv3_relu_bn(low_level_nin, low_level_nout)
        self.conv_out = nn.Conv2d(aspp_nin + low_level_nout, num_classes, 3, bias=True, padding=1)
        
    def forward(self, aspp_out, low_level_features):
        #upsample the aspp outputs by a factor of 4
        out = F.interpolate(aspp_out, scale_factor=4, mode='bilinear', align_corners=True)
        
        #concenate aspp_out with the skip connected low level features
        out = torch.cat([out, self.skip(low_level_features)], dim=1)
        
        #make the prediction
        out = self.conv_out(out)
        
        #return the upsampled prediction
        return F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)  
    
class DeepLabV3(nn.Module):
    """
    Creates the DeepLabV3+ model as presented in:
    https://arxiv.org/abs/1802.02611
    
    Arguments:
    ----------
    n_classes: Int, number of output classes in the segmentation.
    resnet_model: String, choice of [resnet18, resnet34, resnet50]. Default is resnet34. 
    pretrained: Bool. If True, resnet model is initialized with ImageNet pretrained
    weights. If False, resnet model is randomly initialized. Default is True.
    drop_p: Float in [0, 1). Dropout probability applied to both the output of
    the ASPP module and the low level feature map from layer1 of the Encoder's ResNet.
    
    """
    
    def __init__(self, n_classes, resnet_model='resnet34', pretrained=True, drop_p=0):
        super(DeepLabV3, self).__init__()
        
        #create the encoder
        self.encoder = Encoder(resnet_model, pretrained)
        
        #extract all layers prior to the final average pooling
        #and output Linear layer
        aspp_nin = self.encoder.resnet.layer4[-1].conv1.in_channels
        aspp_nout = aspp_nin // 4
        self.aspp = ASPP(aspp_nin, aspp_nout, drop_p)
        
        #create the decoder
        low_level_nin = self.encoder.resnet.layer1[-1].conv1.in_channels
        self.dropout = nn.Dropout2d(p=drop_p)
        self.decoder = Decoder(aspp_nout, low_level_nin, 32, n_classes)
        
    def forward(self, x):
        #run through resnet encoder
        encoder_features = self.encoder(x)
        
        #run the output of the resnet encoder through the aspp layer
        x = self.aspp(self.dropout(encoder_features[-1]))
        
        #apply dropout to the output from layer1 of the encoder resnet
        #and feed it to the decoder along with the output from the aspp module
        low_level_features = self.dropout(encoder_features[1])
        return self.decoder(x, low_level_features)