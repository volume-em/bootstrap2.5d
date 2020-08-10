import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


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
    
class upsample(nn.Module):
    """
    Pytorch layer that applies a 2d transpose convolution with kernel size 2x2 and stride 2. 
    After the conv_transpose layer it applies ReLU activation and batchnorm.
    
    Results in doubling of feature map area, e.g. 14x14 map to 28x28.
    
    Arguments:
    -----------
    
    nin: Number of channels in input, used for Conv2dTranspose
    nout: Desired number of channels in output, used for Conv2dTranspose
    """
    def __init__(self, nin, nout):
        super(upsample, self).__init__()
        self.conv = nn.ConvTranspose2d(nin, nout, 2, stride=2)
        self.bn = nn.BatchNorm2d(nout)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.bn(self.relu(self.conv(x)))
    

    
class skip(nn.Module):
    """
    Pytorch layer that applies the conv1_relu_bn layer to input (see class for documentation).
    
    Prepares downpath to connect with uppath in U-Net. Downsamples feature maps in 
    downpath to match number of feature maps in uppath (not strictly necessary).
    
    Arguments:
    -----------
    
    nin: Number of channels in input, used for Conv2d
    nout: Desired number of channels in output, used for Conv2d
    """
    def __init__(self, nin, nout):
        super(skip, self).__init__()
        self.crb = conv1_relu_bn(nin, nout)
        
    def forward(self, x):
        return self.crb(x)
    
class upblock(nn.Module):
    """
    Pytorch layer the connects downpath and uppath of U-Net.
    
    Applies upsample class on previous step in uppath and concatenates the 
    results with the feature maps from the skip path. 
    
    upd in forward must be exactly half the size of skipd e.g. 
    upd.size = Nx14x14 
    skipd.size = Mx28x28
    M does not necessarily equal N, but it can
    
    Arguments:
    -----------
    
    nin: Number of channels in upd, used for upsample layer
    nout: Desired number of channels in output, used for conv3_relu_bn and upsample
    
    Returns:
    -----------
    
    Given features maps in uppath and downpath it first upsamples uppath to match 
    x,y dimensions of downpath, then concatenates the results and applies a standard
    conv3_bn_relu layer.
    
    Example:
    -----------
    
    upd: (B, nin, 14, 14) ===> (B, nout, 28, 28)
    skipd: (B, nout, 28, 28), in Unet skip path designed to have channel dim nout
    concat: (B, nout * 2, 28, 28)
    out: (B, nout, 28, 28)
    """
    def __init__(self, nin, nout, concat=True):
        super(upblock, self).__init__()
        self.concat = concat
        self.up = upsample(nin, nout)
        
        if concat:
            self.crb1 = conv3_relu_bn(nout * 2, nout)
        else:
            self.crb1 = conv3_relu_bn(nout, nout)
        #self.crb2 = conv3_relu_bn(nout, nout)

    def forward(self, upd, skipd):
        x = self.up(upd)
        if self.concat:
            xs = self.crb1(torch.cat([x, skipd], dim=1))
        else:
            xs = self.crb1(x + skipd)
        
        return xs
    
class lastblock(nn.Module):
    """
    Pytorch layer that produces the final mask for U-net.
    
    Applies upsample class on previous step in uppath and concatenates the 
    results with feature maps derived from the full size input image. 
    
    upd in forward must be exactly half the size of skipd e.g.: 
    upd.size = Nx14x14 
    skipd.size = Mx28x28
    M does not necessarily equal N, but it can
    
    Arguments:
    -------------
    
    nin: Number of channels in upd, used for upsample layer
    nout: Desired number of channels in output, used for Conv2d on input image
    and upsample operation on uppath
    
    Returns:
    ------------
    
    Predicted mask for input image. Size of predicted mask is (B, 1, img_size) where
    img_size are the y,x dimensions of the input image.
    
    NOTE: Pytorch recommends using BCEWithLogitsLoss over BCELoss. BCEWithLogitsLoss
    includes a Sigmoid activation, which is why the output of this layer does not 
    include that operation.
    
    Example:
    ------------
    
    upd: (B, nin, 112, 112) ===> (B, nout, 224, 224)
    skipd: (B, 1, 224, 224) ===> (B, nout, 224, 224)
    concat: (B, nout * 2, 224, 224)
    out: (B, 1, 224, 224)
    """
    def __init__(self, nin, nout, concat=True):
        super(lastblock, self).__init__()
        self.concat = concat
        self.up = upsample(nin, nout)
        self.conv = nn.Conv2d(1, nout, 3, 1, padding=1)
        self.bn = nn.BatchNorm2d(nout)
        self.relu = nn.ReLU(inplace=True)
        
        if concat:
            self.conv_out = nn.Conv2d(nout * 2, 1, 3, 1, padding=1)
        else:
            self.conv_out = nn.Conv2d(nout, 1, 3, 1, padding=1)
        
    def forward(self, upd, skipd):
        x = self.up(upd)
        s = self.bn(self.relu(self.conv(skipd)))
        if self.concat:
            xs = self.conv_out(torch.cat([x, s], dim=1))
        else:
            xs = self.conv_out(x + s)
        
        return xs
    
    
class UResNet(nn.Module):
    """
    Class for creating a UNet with abitrary ResNet backbone
    
    backbone: A ResNet model from torchvision.models, preferably pretrained.
    concat: If concat is True, the skip connection from the downpath and channels from uppath
    are concatenated together. If False, they are added. Default is True.
    
    Returns:
    ---------------
    
    A Pytorch model
    """
    
    def __init__(self, backbone, concat=True):
        """
        Backbone is the pretrained ResNet Model
        """
        super(UResNet, self).__init__()
        
        self.down1 = nn.Sequential(backbone.conv1,
                                   backbone.bn1,
                                   backbone.relu,
                                   backbone.maxpool,
                                   backbone.layer1) #112x112
        self.down2 = backbone.layer2 #56x56
        self.down3 = backbone.layer3 #28x28
        self.down4 = backbone.layer4 #14x14
        
        self.up1 = upblock(512, 256, concat=concat) #28x28
        self.up2 = upblock(256, 128, concat=concat) #56x56
        self.up3 = upblock(128, 64, concat=concat) #112x112
        self.out_block = lastblock(64, 32, concat=concat)
        
    def forward(self, x):
        #down path
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        #up path and skips
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        
        return self.out_block(u3, x)
    
    
class ImagePooling(nn.Module):
    def __init__(self, nin, nout, pool_size):
        super(ImagePooling, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.conv = conv1_relu_bn(nin, nout)
        
    def forward(self, x):
        unpool_size = (x.size(2), x.size(3)) #get height and width
        x = self.pool(x)
        x = self.conv(x)
        
        return F.interpolate(x, size=unpool_size, mode='bilinear', align_corners=True)
    
class ASPP(nn.Module):
    def __init__(self, nin, nout_each, drop_p=0, dilations=[2, 4, 6]):
        super(ASPP, self).__init__()
        self.conv1 = conv1_relu_bn(nin, nout_each)
        
        d1, d2, d3 = dilations
        self.atrous2 = conv3_relu_bn(nin, nout_each, padding=d1, dilation=d1)
        self.atrous4 = conv3_relu_bn(nin, nout_each, padding=d2, dilation=d2)
        self.atrous8 = conv3_relu_bn(nin, nout_each, padding=d3, dilation=d3)
        self.im_pool = ImagePooling(nin, nout_each, pool_size=1)
        
        self.reduce = conv3_relu_bn(nout_each * 5, nout_each, 1, 1)
        self.drop = nn.Dropout2d(drop_p)
        
    def forward(self, x):
        pyros = []
        pyros.append(self.conv1(x))
        pyros.append(self.atrous2(x))
        pyros.append(self.atrous4(x))
        pyros.append(self.atrous8(x))
        pyros.append(self.im_pool(x))
        
        x = torch.cat(pyros, dim=1)
        x = self.reduce(x)
        
        return self.drop(x)
    
class Decoder(nn.Module):
    def __init__(self, aspp_nin=64, low_level_nin=24, low_level_nout=32, num_classes=1):
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
    Class for creating a UNet with abitrary ResNet backbone
    
    backbone: A ResNet model from torchvision.models, preferably pretrained.
    concat: If concat is True, the skip connection from the downpath and channels from uppath
    are concatenated together. If False, they are added. Default is True.
    
    Returns:
    ---------------
    
    A Pytorch model
    """
    
    def __init__(self, backbone, n_classes, model_type='resnet', drop_p=0):
        """
        Backbone is the pretrained ResNet Model
        """
        super(DeepLabV3, self).__init__()
        #change stride in backbone layer4 to 1
        if model_type == 'resnet':
            prefix = backbone
        elif model_type == 'insresnet' or model_type == 'insresnet_aspp':
            prefix = backbone.encoder.module
        else:
            raise Exception("Model type {} not recognized".format(model_type))
            
        #there are two possibilities, either conv1 or conv2
        #will have a stride of 2, (e.g. for resnet 34 vs resnet50)
        l4_c1_stride = prefix.layer4[0].conv1.stride
        if l4_c1_stride == (2, 2):
            prefix.layer4[0].conv1.stride = (1, 1)
        else:
            prefix.layer4[0].conv2.stride = (1, 1)

        #the downsample layer always has stride (2, 2)
        prefix.layer4[0].downsample[0].stride = (1, 1)
        
        #register a forward hook on the intermediate output
        prefix.layer1[-1].register_forward_hook(self.hook)
        
        #extract all layers prior to the final average pooling
        #and output Linear layer
        if model_type == 'resnet' or model_type == 'resnext':
            self.backbone = nn.Sequential(*list(prefix.children()))[:-2]
            aspp_nin = prefix.layer4[-1].conv1.in_channels
            aspp_nout = aspp_nin // 4
            self.aspp = ASPP(aspp_nin, aspp_nout, drop_p)
        elif model_type == 'insresnet':
            self.backbone = nn.Sequential(*list(prefix.children()))[:-3]
            aspp_nin = prefix.layer4[-1].conv1.in_channels
            aspp_nout = aspp_nin // 4
            self.aspp = ASPP(aspp_nin, aspp_nout, drop_p)
        elif model_type == 'insresnet_aspp':
            self.backbone = nn.Sequential(*list(prefix.children()))[:-4]
            self.aspp = prefix.aspp
            self.aspp.drop.p = drop_p
            aspp_nout = prefix.aspp.reduce.out_channels
        else:
            raise Exception("Model type {} not recognized".format(model_type))
        
        #define the decoder
        low_level_nin = prefix.layer1[-1].conv1.in_channels
        self.decoder = Decoder(aspp_nout, low_level_nin, 32, n_classes)
        self.dropout = nn.Dropout2d(p=drop_p)
        
    def hook(self, in1, in2, output):
        #forward hook receives batchnorm parameters
        #for in1, in2, features we want are in output
        self.intermediate = output
        
    def forward(self, x):
        #run through resnet backbone
        x = self.dropout(self.backbone(x))
        #run through aspp
        x = self.aspp(x)
        #decode and upsample for the output
        low_level_features = self.dropout(self.intermediate)
        return self.decoder(x, low_level_features)
    
class MobileDeepLab(nn.Module):
    def __init__(self, num_classes, drop_p=0, weight_path=None):
        super(MobileDeepLab, self).__init__()
        #to use mobile net, change the stride for the last
        #strided convolution from (2, 2) to (1, 1) this
        #make the output scale 16 instead of 32
        backbone = mobilenet_v2(pretrained=True)
        backbone.features[14].conv[1][0].stride = (1, 1)
        #register a hook to get low level features
        backbone.features[3].conv[-1].register_forward_hook(self.hook)
        #set the backbone
        self.backbone = nn.Sequential(*list(backbone.features.children())[:-1])
        #create aspp heads
        self.aspp = ASPP(320, 64)
        #make the decoder
        self.decoder = Decoder(320 // 5, 24, num_classes, drop_p=drop_p)
    
    def hook(self, in1, in2, output):
        #forward hook receives batchnorm parameters
        #for in1, in2, features we want are in output
        self.output = output
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.aspp(x)
        return self.decoder(x, self.output)
    
