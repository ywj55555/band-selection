import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict

def conv3x3(in_channels, out_channels, stride=1): #kernel_size为3的不加偏置的卷积层
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1,padding_mode='reflect',bias=False)

class Mish(nn.Module): #一种高效的激活函数
    def __init__(self):
        super().__init__()
        # print("Mish activation loaded...")
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

def l1_regularization(model, l1_alpha):
    l1_loss = []
    for module in model.modules():
        if type(module) is nn.BatchNorm2d:
            l1_loss.append(torch.abs(module.weight).sum())
    return l1_alpha * sum(l1_loss)

def l2_regularization(model, l2_alpha=0.01):#alpha=0.01,再进一步地细调节，比如调节为0.02，0.03，0.009之类
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)

class SELayer(nn.Module): #初始通道注意力
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel*reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel*reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ywjCSLayer(nn.Module): #空间、通道注意力
    def __init__(self, channel, ratio=3):
        super(ywjCSLayer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channel,channel*ratio,kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(channel*ratio, channel, kernel_size=1, stride=1)
        self.mish = Mish()
        # self.fc = nn.Sequential(
        #     nn.Linear(channel, channel*reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel*reduction, channel, bias=False),
        #     nn.Sigmoid()
        # )
    def forward(self, x):
        # b, c, h, w = x.size()
        y=self.mish(self.conv1(x))
        y2=torch.sigmoid(self.conv2(y))
        # y = self.avg_pool(x).view(b, c)
        # y = self.fc(y).view(b, c, 1, 1)
        return x * y2


class ECAlayer(nn.Module): #一维卷积通道注意力，适用于通道数很多的注意力，较少运算
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ECAlayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SpatialAttention(nn.Module):#初始空间注意力，取均值和max cat再卷积
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):# 取均值和max相加的通道注意力
    def __init__(self, in_planes, ratio=3):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes*ratio, 1, bias=False)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.fc2 = nn.Conv2d(in_planes*ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class ConvBN(nn.Sequential):    #卷积块，加了BN层后，就不用加偏置了
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding,padding_mode='reflect',groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes)),
            #('Mish', Mish())
            ('LeakyRelu', nn.LeakyReLU(negative_slope=0.3, inplace=False))
        ]))

class ConvBNRe(nn.Sequential):    #卷积块，加了BN层后，就不用加偏置了
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,padding=0,groups=1):
        # if not isinstance(kernel_size, int):
        #     padding = [(i - 1) // 2 for i in kernel_size]
        # else:
        #     padding = (kernel_size - 1) // 2
        super(ConvBNRe, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding,padding_mode='reflect',groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes)),
            #('Mish', Mish())
            ('Relu', nn.ReLU(inplace=True))
        ]))

class deConvBNRe(nn.Sequential):    #卷积块，加了BN层后，就不用加偏置了
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,padding=0,groups=1):
        # if not isinstance(kernel_size, int):
        #     padding = [(i - 1) // 2 for i in kernel_size]
        # else:
        #     padding = (kernel_size - 1) // 2
        super(deConvBNRe, self).__init__(OrderedDict([
            ('conv', nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding,groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes)),
            #('Mish', Mish())
            ('Relu', nn.ReLU(inplace=True))
        ]))

class ConvBNNP(nn.Sequential):    #卷积块，加了BN层后，就不用加偏置了
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        super(ConvBNNP, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                               groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes)),
            #('Mish', Mish())
            ('LeakyRelu', nn.LeakyReLU(negative_slope=0.3, inplace=False))
        ]))


class Conv3BN(nn.Sequential):
    def __init__(self, in_planes, ratio=3, kernel_size=3, stride=1, groups=1):
        # self.base=Conv3BN(in_planes,in_planes,kernel_size,stride,groups)
        super(Conv3BN, self).__init__(OrderedDict([
            ('ConvBNNP1',ConvBNNP(in_planes,in_planes*ratio,kernel_size,stride,groups)),
            ('ConvBNNP2', ConvBNNP(in_planes*ratio, in_planes*ratio, kernel_size, stride, groups)),
            ('ConvBNNP3', ConvBNNP(in_planes*ratio, in_planes, kernel_size, stride, groups)),

        ]))

# class numCBN(nn.Sequential):
#     def __init__(self, in_planes,num=5,kernel_size=3, stride=1, groups=1):
#         # self.base=Conv3BN(in_planes,in_planes,kernel_size,stride,groups)
#         super(numCBN, self).__init__(OrderedDict([
#             ('ConvBNNP'+str(i+1),ConvBNNP(in_planes,in_planes,kernel_size,stride,groups)) for i in range(num)
#         ]))

class Conv4BN(nn.Sequential):
    def __init__(self, in_planes, ratio=3, kernel_size=3, stride=1, groups=1):
        # self.base=Conv3BN(in_planes,in_planes,kernel_size,stride,groups)
        super(Conv4BN, self).__init__(OrderedDict([
            ('ConvBNNP1',ConvBNNP(in_planes,in_planes,kernel_size,stride,groups)),
            ('ConvBNNP2', ConvBNNP(in_planes, in_planes*ratio, kernel_size, stride, groups)),
            ('ConvBNNP3', ConvBNNP(in_planes*ratio, in_planes*ratio, kernel_size, stride, groups)),
            ('ConvBNNP4', ConvBNNP(in_planes * ratio, in_planes, kernel_size, stride, groups)),
            # ('ConvBNNP5', ConvBNNP(in_planes, in_planes, kernel_size, stride, groups)),
        ]))

class ResBlock(nn.Module): #初始残差块结构
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self,ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(ConvBN(ch, ch, 3))
            resblock_one.append(Mish())
            resblock_one.append(ConvBN(ch, ch, 3))
            resblock_one.append(Mish())
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x

class BasicResBlock(nn.Module):#另一种残差块结构，和上面一样kernel——size不太一样而已，上面是1和3，这个是两个3，激活函数要不要换一下
    def __init__(self, inplanes, planes, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

class CifarSEBasicBlock(nn.Module):#通道注意力的残差块
    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(CifarSEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride
    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out

class CSResBlock(nn.Module): #两个卷积后使用通道和空间注意力，最后残差连接，就是加了通道注意力和空间注意力的残差块
    def __init__(self, inplanes, planes, stride=1):
        super(CSResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.ca = ChannelAttention(planes,3)
        self.sa = SpatialAttention()
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride
    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out # 广播机制
        out = self.sa(out) * out # 广播机制
        # if self.downsample is not None:
        #     residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class SResBlock(nn.Module): #两个卷积后使用空间注意力，最后残差连接，就是加了空间注意力的残差块，分割图训练可能不适用通道注意力
    def __init__(self, inplanes, planes, stride=1):
        super(CSResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.ca = ChannelAttention(planes,3)
        self.sa = SpatialAttention()
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride
    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.ca(out) * out # 广播机制
        out = self.sa(out) * out # 广播机制
        # if self.downsample is not None:
        #     residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
# class res18(nn.Module):#dual pooling
#     def __init__(self, num_classes):
#         super(res18, self).__init__()
#         self.base = resnet18(pretrained=True)
#         self.feature = nn.Sequential(
#         self.base.conv1,
#         self.base.bn1,
#         self.base.relu,
#         self.base.maxpool,
#         self.base.layer1,
#         self.base.layer2,
#         self.base.layer3,
#         self.base.layer4
#         )
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.reduce_layer = nn.Conv2d(1024, 512, 1)
#         self.fc = nn.Sequential(
#         nn.Dropout(0.5),
#         nn.Linear(512, num_classes)
#         )
#     def forward(self, x):
#         bs = x.shape[0]
#         x = self.feature(x)
#         x1 = self.avg_pool(x)
#         x2 = self.max_pool(x)
#         x = torch.cat([x1, x2], dim=1)
#         x = self.reduce_layer(x).view(bs, -1)
#         logits = self.fc(x)
#         return logits

class CA_Block(nn.Module): #Coordinate Attention，主要也是通道注意力，分成高、宽方向的平均
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out


