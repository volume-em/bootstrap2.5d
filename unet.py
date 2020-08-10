import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class conv5_relu_bn(nn.Module):
    """
    Pytorch layer that applies a 2d convolution with kernel size 5x5, stride 1
    and padding 1. After the conv layer it applies ReLU activation and batchnorm.
    
    Arguments:
    -----------
    
    nin: Number of channels in input, used for Conv2d
    nout: Desired number of channels in output, used for Conv2d
    """
    def __init__(self, nin, nout, padding=2, dilation=1):
        super(conv5_relu_bn, self).__init__()
        self.conv = nn.Conv2d(nin, nout, 5, 1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(nout)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.bn(self.relu(self.conv(x)))
    
class Encoder(nn.Module):
    def __init__(self, backbone):
        """
        Backbone encoder for deeplab
        """
        super(Encoder, self).__init__()
        
        #there are two possibilities, either conv1 or conv2
        #will have a stride of 2, (e.g. for resnet 34 vs resnet50)
        #l4_c1_stride = backbone.layer4[0].conv1.stride
        #if l4_c1_stride == (2, 2):
        #    backbone.layer4[0].conv1.stride = (1, 1)
        #else:
        #    backbone.layer4[0].conv2.stride = (1, 1)

        #the downsample layer always has stride (2, 2)
        #backbone.layer4[0].downsample[0].stride = (1, 1)
        
        #extract all layers prior to the final average pooling
        #and output Linear layer
        self.backbone = backbone
        
        self.groups = [
            nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu), #112, 112
            nn.Sequential(self.backbone.maxpool, self.backbone.layer1), #56, 56
            self.backbone.layer2, #28, 28
            self.backbone.layer3, #14, 14
            self.backbone.layer4 #7, 7
        ]
        
    def forward(self, x):
        features = []
        for group in self.groups:
            x = group(x)
            features.append(x)
        
        return features
    
class decoder_block(nn.Module):
    def __init__(self, nin, skip_nin, nout, scale_factor=2):
        super(decoder_block, self).__init__()
        
        #define the skip connection
        #self.skip = conv3_relu_bn(skip_nin, skip_nin)
        
        #define the upsampling block as 3x3 conv
        self.upconv = conv3_relu_bn(nin, skip_nin)
        
        #define the concat convolution
        self.conv1 = conv3_relu_bn(skip_nin * 2, nout)
        #self.conv2 = conv3_relu_bn(nout, nout)
        self.scale_factor = scale_factor
        
    def forward(self, x, skip):
        #skip = self.skip(skip)
        x = self.upconv(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        return self.conv1(torch.cat([x, skip], dim=1))
        
    
class Decoder(nn.Module):
    def __init__(self, encoder, num_classes=1):
        super(Decoder, self).__init__()
            
        #in the paper they reduce the number of channels by the reduction factor
        block1_skip_nin = encoder.groups[-2][-1].conv1.in_channels #1024
        block1_nin = encoder.groups[-1][-1].conv1.in_channels #2048
        block1_nout = block1_skip_nin
        self.block1 = decoder_block(block1_nin, block1_skip_nin, block1_nout, scale_factor=2)
        
        block2_skip_nin = encoder.groups[-3][-1].conv1.in_channels #512
        block2_nin = block1_nout #1024
        block2_nout = block2_skip_nin
        self.block2 = decoder_block(block2_nin, block2_skip_nin, block2_nout, scale_factor=2)
        
        block3_skip_nin = encoder.groups[-4][-1][-1].conv1.in_channels #256
        block3_nin = block2_nout #512
        block3_nout = block3_skip_nin
        self.block3 = decoder_block(block3_nin, block3_skip_nin, block3_nout, scale_factor=2)
        
        block4_skip_nin = encoder.groups[-5][0].out_channels #64
        block4_nin = block3_nout #256
        block4_nout = block4_skip_nin
        self.block4 = decoder_block(block4_nin, block4_skip_nin, block4_nout, scale_factor=2)
        
    def forward(self, features):
        x = self.block1(features[-1], features[-2])
        x = self.block2(x, features[-3])
        x = self.block3(x, features[-4])
        x = self.block4(x, features[-5])
        
        return x
    
class SegmentationHead(nn.Module):
    def __init__(self, nin, n_classes):
        super(SemanticHead, self).__init__()
        self.conv5 = conv5_relu_bn(nin, nin)
        self.conv1 = nn.Conv2d(nin, n_classes, 1, bias=True)
        
    def forward(self, x):
        return self.conv1(self.conv5(x))
    
class UNet(nn.Module):
    """
    Class for creating a UNet with abitrary ResNet backbone
    
    backbone: A ResNet model from torchvision.models, preferably pretrained.
    concat: If concat is True, the skip connection from the downpath and channels from uppath
    are concatenated together. If False, they are added. Default is True.
    
    Returns:
    ---------------
    
    A Pytorch model
    """
    
    def __init__(self, backbone, num_classes=1):
        """
        Backbone is the pretrained ResNet Model
        """
        super(UNet, self).__init__()
        #change stride in backbone layer4 to 1
        #create the encoder
        self.encoder = Encoder(backbone)
        self.decoder = Decoder(self.encoder)
        
    def forward(self, x):
        features = self.encoder(x)
        
        return self.decoder(features)
