import torch.nn as nn
from cnn_block import ConvBNRe,deConvBNRe
from materialNet import MaterialSubModel
from model.DBDA_Conv import DBDA_network_MISH_full_conv, DBDA_network_MISH_without_bn, DBDA_network_MISH
import torch
class BSNET_Conv(nn.Module):
    def __init__(self,bands):
        super(BSNET_Conv, self).__init__()
        #BAM
        self.bands = bands
        self.bn1 = nn.Sequential(nn.BatchNorm2d(self.bands))
        self.conv1 = ConvBNRe(self.bands,64,kernel_size=3, stride=1)
            # nn.Sequential(
            # nn.Conv2d(200, 64, (3, 3), 1, 0),
            # nn.ReLU(True))
        self.gp = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Sequential(
            nn.Linear(64, 96),
            nn.ReLU(True))

        self.fc2 = nn.Sequential(
            nn.Linear(96, self.bands),
            nn.Sigmoid())

        #reconstruction
        self.conv1_1 = ConvBNRe(bands,96,kernel_size=3, stride=1)
            # nn.Sequential(
            # nn.Conv2d(200, 128, (3, 3), 1, 0),
            # nn.ReLU(True))
        self.conv1_2 = ConvBNRe(96,64,kernel_size=3, stride=1)
            # nn.Sequential(
            # nn.Conv2d(128, 64, (3, 3), 1, 0),
            # nn.ReLU(True))

        self.deconv1_2 = deConvBNRe(64,64,kernel_size=3, stride=1)
            # nn.Sequential(
            # nn.ConvTranspose2d(64, 64, (3, 3), 1, 0),
            # nn.ReLU(True))

        self.deconv1_1 = deConvBNRe(64,96,kernel_size=3, stride=1)
            # nn.Sequential(
            # nn.ConvTranspose2d(64, 128, (3, 3), 1, 0),
            # nn.ReLU(True))

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(96, self.bands, (1, 1), 1, 0),
            nn.Sigmoid())

    def BAM(self, x):#计算的就是weights
        x = self.conv1(x)
        x = self.gp(x)
        x = x.view(-1, 64)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 1, 1, self.bands)
        x = x.permute(0, 3, 1, 2)
        return x

    def RecNet(self, x):
        # 下采样
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        # 上采样
        x = self.deconv1_2(x)
        x = self.deconv1_1(x)
        # 1 * 1 调整一下
        x = self.conv2_1(x)
        return x

    def forward(self, x):
        # 一输入进来就 BN 层， 应该就不用 通道归一化了？ 原始论文 代码也是这样
        x = self.bn1(x)
        # 求 各个 通道的 权重
        BRW = self.BAM(x)
        x = x * BRW
        # 重构网络 换成 分类网络就好了？
        ret = self.RecNet(x)
        return ret,BRW #B,self.bands,1,1 还要参与loss的计算
    #可以返回两个值，然后一个用于loss计算，另一个不参与计算吗？BRW激活值为0-1之间

class BSNET_Embedding(nn.Module):
    def __init__(self, bands, class_nums, class_model, depth_channel=False):
        super(BSNET_Embedding, self).__init__()
        self.bands = bands
        self.class_nums = class_nums
        self.depth_channel = depth_channel
        #BAM
        self.bn1 = nn.Sequential(nn.BatchNorm2d(self.bands))
        self.conv1 = ConvBNRe(self.bands,64,kernel_size=3, stride=1)
        self.gp = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(64, 96),
            nn.ReLU(True))
        self.fc2 = nn.Sequential(
            nn.Linear(96, self.bands),
            nn.Sigmoid())

        #classification
        self.classification = class_model(self.bands, self.class_nums)

    # 计算的就是weights
    def BAM(self, x):
        x = self.conv1(x) #最好是用两个卷积把！
        x = self.gp(x)
        x = x.view(-1, 64)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 1, 1, self.bands)
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        # 一输入进来就 BN 层， 应该就不用 通道归一化了？ 原始论文 代码也是这样
        x = self.bn1(x)
        # 求 各个 通道的 权重
        BRW = self.BAM(x)
        x = x * BRW
        # 重构网络 换成 分类网络就好了？
        if self.depth_channel:
            # B D H W -> B C H W D
            x = x.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
        pred = self.classification(x)
        return pred, BRW  # B,self.bands,1,1 还要参与loss的计算
    #可以返回两个值，然后一个用于loss计算，另一个不参与计算吗？BRW激活值为0-1之间

if __name__ == '__main__':
    img = torch.rand((4, 128, 11, 11)).cuda()
    # model1 = BSNET_Embedding(128, 2, MaterialSubModel, False).cuda()
    model2 = BSNET_Embedding(128, 2, DBDA_network_MISH, True).cuda()
    # output1 = model1(img)
    output2 = model2(img)
    # print(output1[0].shape)
    print(output2[0].shape)

