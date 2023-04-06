import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.utilNetNew_old2 import *


# 问问高斯核怎么弄
class Mish(nn.Module): #一种高效的激活函数
    def __init__(self):
        super().__init__()
        # print("Mish activation loaded...")
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, inputData, inputLabel):
        self.Data = inputData
        self.Label = inputLabel

    def __getitem__(self, index):
        img = self.Data[index]
        label = self.Label[index]
        return img, label

    def __len__(self):
        return len(self.Data)

class MyDatasetAddSpac(torch.utils.data.Dataset):#看下数据处理步骤
    def __init__(self,inputData,inputNorData,inputLabel):
        self.Data=inputData
        self.NorData=inputNorData
        self.Label=inputLabel
    def __getitem__(self, index): #实现__getitem__和__len__就好了，注意对应返回的值 训练数据、标签
        img=self.Data[index]
        nor_img=self.NorData[index]
        label=self.Label[index]
        return img,nor_img,label
    def __len__(self):
        return len(self.Data)

class MaterialSubModel(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels_1=16, mid_channels_2=32, mid_channels_3=8):
        super(MaterialSubModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels_1, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU()
            #nn.ReLU()
            #nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(mid_channels_1, mid_channels_2, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU()
            # nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(mid_channels_2, mid_channels_3, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU()
            # nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(mid_channels_3, out_channels, kernel_size=3, stride=1, padding=0),
            #网络的最后一层最好不用relu激活，一般分类问题用softmax激活
            nn.LeakyReLU()
            # nn.ReLU()
        )

    def forward(self, x):
        x=self.layer1(x)
        #print('layer1:',x.size())
        x=self.layer2(x)
        #print('layer2:', x.size())
        x = self.layer3(x)
        #print('layer3:', x.size())
        x = self.layer4(x)
        #x = x.view(x.size(0),-1)
#        print('layer4:', x.size())
        #x=self.linear(x)
        return x


class BasicMaterial(nn.Module):
    def __init__(self, in_channels, out_channels,expansion1=2,expansion2=2):
        super(BasicMaterial, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*expansion1, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU()
            #nn.ReLU()
            #nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels*expansion1, in_channels*expansion1, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU()
            # nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels*expansion1, out_channels*expansion2, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU()
            # nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(out_channels*expansion2, out_channels, kernel_size=3, stride=1, padding=0),
            #网络的最后一层最好不用relu激活，一般分类问题用softmax激活
            # softmax可以不加激活函数
            # nn.LeakyReLU()
            # nn.ReLU()
        )

    def forward(self, x):
        x=self.layer1(x)
        #print('layer1:',x.size())
        x=self.layer2(x)
        #print('layer2:', x.size())
        x = self.layer3(x)
        #print('layer3:', x.size())
        x = self.layer4(x)
        #x = x.view(x.size(0),-1)
#        print('layer4:', x.size())
        #x=self.linear(x)
        return x

class TwoStreamMaterial(nn.Module):
    def __init__(self,fir_channels=18, sec_channels=3, out_channels=3):
        super(TwoStreamMaterial, self).__init__()
        self.spec_stream = BasicMaterial(fir_channels,out_channels,2,2)

        self.yuv_stream = BasicMaterial(sec_channels, out_channels,6,2)
        # 拼接最好使用通道注意力机制，或者分别输出18通道和3通道
        self.fina_layer = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=1, stride=1, padding=0),
            #网络的最后一层最好不用relu激活，一般分类问题用softmax激活
            # nn.LeakyReLU(),
            # 使用nn.CrossEntropyLoss() 最后一层就不用激活了！！
            # nn.Softmax(dim=1)
            # nn.ReLU()
        )

    def forward(self,spec_data,yuv_data):
        y1 = self.spec_stream(spec_data)
        # print('layer1:',x.size())
        y2 = self.yuv_stream(yuv_data)
        # print('layer2:', x.size())
        y = torch.cat([y1,y2],dim=1)
        res = self.fina_layer(y)
        # x = self.layer3(x)
        # print('layer3:', x.size())
        # x = self.layer4(x)
        # x = x.view(x.size(0),-1)
        #        print('layer4:', x.size())
        # x=self.linear(x)
        return res

class TwoStreamMaterialDifCha(nn.Module):
    def __init__(self,fir_channels=18, sec_channels=3, out_channels=3):
        super(TwoStreamMaterialDifCha, self).__init__()
        self.spec_stream = nn.Sequential(
            nn.Conv2d(fir_channels, fir_channels*2, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(fir_channels*2, fir_channels * 2, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(fir_channels*2, fir_channels , kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(fir_channels, fir_channels, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU()
            #nn.ReLU()
            #nn.ReLU(inplace=True)
        )

        self.yuv_stream = nn.Sequential(
            nn.Conv2d(sec_channels, sec_channels * 6, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(sec_channels * 6, sec_channels * 4, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(sec_channels * 4, sec_channels*2, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(sec_channels*2, sec_channels, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU()
            #nn.ReLU()
            #nn.ReLU(inplace=True)
        )
        # 拼接最好使用通道注意力机制，或者分别输出18通道和3通道
        self.fina_layer = nn.Sequential(
            nn.Conv2d(fir_channels+sec_channels, out_channels, kernel_size=1, stride=1, padding=0),
            #网络的最后一层最好不用relu激活，一般分类问题用softmax激活
            # nn.LeakyReLU(),
            # 使用nn.CrossEntropyLoss() 最后一层就不用激活了！！
            # nn.Softmax(dim=1)
            # nn.ReLU()
        )

    def forward(self,spec_data,yuv_data):
        y1 = self.spec_stream(spec_data)
        # print('layer1:',x.size())
        y2 = self.yuv_stream(yuv_data)
        # print('layer2:', x.size())
        y = torch.cat([y1,y2],dim=1)
        res = self.fina_layer(y)
        # x = self.layer3(x)
        # print('layer3:', x.size())
        # x = self.layer4(x)
        # x = x.view(x.size(0),-1)
        #        print('layer4:', x.size())
        # x=self.linear(x)
        return res

class SELayer(nn.Module): #
    def __init__(self, channel, ratio=3):
        super(SELayer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel * ratio, kernel_size=1, stride=1),
            # nn.LeakyReLU(),
            Mish()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel* ratio, channel, kernel_size=1, stride=1),
        )

        # self.fc = nn.Sequential(
        #     nn.Linear(channel, channel*reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel*reduction, channel, bias=False),
        #     nn.Sigmoid()
        # )
    def forward(self, x):
        y = self.conv1(x)
        y2 = torch.sigmoid(self.conv2(y))

        # b, c, _, _ = x.size()
        # y = self.avg_pool(x).view(b, c)
        # y = self.fc(y).view(b, c, 1, 1)
        return x * y2

class TwoStreamMaterialChaAtt(nn.Module):
    def __init__(self, fir_channels=18, sec_channels=3, out_channels=3):
        super(TwoStreamMaterialChaAtt, self).__init__()
        self.spec_stream = BasicMaterial(fir_channels, out_channels, 2, 2)
        # 拼接最好使用通道注意力机制，或者分别输出18通道和3通道  两个模型都试试！！！

        self.yuv_stream = BasicMaterial(sec_channels, out_channels, 6, 2)
        self.fina_layer = nn.Sequential(
            SELayer(out_channels * 2),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0),
            # 网络的最后一层最好不用relu激活，一般分类问题用softmax激活
            # nn.LeakyReLU(),
            # 使用nn.CrossEntropyLoss() 最后一层就不用激活了！！
            # nn.Softmax(dim=1)
            # nn.ReLU()
        )

    def forward(self, spec_data, yuv_data):
        y1 = self.spec_stream(spec_data)
        # print('layer1:',x.size())
        y2 = self.yuv_stream(yuv_data)
        # print('layer2:', x.size())
        y = torch.cat([y1, y2], dim=1)
        res = self.fina_layer(y)
        # x = self.layer3(x)
        # print('layer3:', x.size())
        # x = self.layer4(x)
        # x = x.view(x.size(0),-1)
        #        print('layer4:', x.size())
        # x=self.linear(x)
        return res

class TwoStreamMiddMaterial(nn.Module):
    def __init__(self,fir_channels=18, sec_channels=3, out_channels=3):
        super(TwoStreamMiddMaterial, self).__init__()
        self.spec_stream = nn.Sequential(
            nn.Conv2d(fir_channels, fir_channels*2, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(fir_channels*2, fir_channels*2, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(fir_channels*2, fir_channels, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU())

        self.yuv_stream = nn.Sequential(
            nn.Conv2d(sec_channels, sec_channels*6, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(sec_channels*6, sec_channels*6, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(sec_channels * 6, sec_channels * 3, kernel_size=3, stride=1,
                      padding=0),
            nn.LeakyReLU()
            )

        self.fina_layer = nn.Sequential(
            nn.Conv2d(fir_channels + sec_channels * 3, out_channels*3, kernel_size=3, stride=1, padding=0),
            #网络的最后一层最好不用relu激活，一般分类问题用softmax激活
            nn.LeakyReLU(),
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, stride=1, padding=0),
            # 使用nn.CrossEntropyLoss() 最后一层就不用激活了！！
            # nn.Softmax(dim=1)
            # nn.ReLU()
        )

    def forward(self,spec_data,yuv_data):
        y1 = self.spec_stream(spec_data)
        # print('layer1:',x.size())
        y2 = self.yuv_stream(yuv_data)
        # print('layer2:', x.size())
        y = torch.cat([y1,y2],dim=1)
        res = self.fina_layer(y)
        return res

class CatMaterial(nn.Module):
    def __init__(self, in_channels=21, out_channels=3,expansion1=2,expansion2=2):
        super(CatMaterial, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*expansion1, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU()
            #nn.ReLU()
            #nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels*expansion1, in_channels*expansion1, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU()
            # nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels*expansion1, out_channels*expansion2, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU()
            # nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(out_channels*expansion2, out_channels, kernel_size=3, stride=1, padding=0),
            #网络的最后一层最好不用relu激活，一般分类问题用softmax激活
            # nn.LeakyReLU()
            # nn.ReLU()
        )

    def forward(self, x):
        x=self.layer1(x)
        #print('layer1:',x.size())
        x=self.layer2(x)
        #print('layer2:', x.size())
        x = self.layer3(x)
        #print('layer3:', x.size())
        x = self.layer4(x)
        #x = x.view(x.size(0),-1)
#        print('layer4:', x.size())
        #x=self.linear(x)
        return x

class MaterialModel(nn.Module):
    def __init__(self):
        super(MaterialModel, self).__init__()
        self.subModel_skin = MaterialSubModel(8, 2)
        self.subModel_cloth = MaterialSubModel(6, 2)
        self.subModel_plant = MaterialSubModel(6, 2)
        self.linear = nn.Linear(6, 4)

    def forward(self, x):
        x1 = x[:, 0:8, :, :]
        x1 = self.subModel_skin(x1)
        #print(x1.size())
        x2 = x[:, 8:14, :, :]
        x2 = self.subModel_cloth(x2)
        #print(x2.size())
        x3 = x[:, 14:20, :, :]
        x3 = self.subModel_plant(x3)
        #print(x3.size())
        x = torch.cat([x1, x2, x3], 1)
        x = x.squeeze()
        x = x.squeeze()
        #print(x.size())
        x = self.linear(x)
        return x


class channellModel(nn.Module):
    def __init__(self,in_channels=9,mid_channels_1=32,mid_channels_2=64,out_channels=9):
        super(channellModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels_1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels_1),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(mid_channels_1, mid_channels_2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels_2),
            nn.LeakyReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(mid_channels_2, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class spacialModel(nn.Module):
    def __init__(self,in_channels=19,mid_channels_1=32,out_channels=9):
        super(spacialModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels_1, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels_1),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(mid_channels_1, out_channels, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
class materialModel(nn.Module):
    def __init__(self):
        super(materialModel, self).__init__()
        self.channelModel = channellModel(9,32,64,9)
        self.spaceModel = spacialModel(19+9,32,4)
    def forward(self, x):
        x1 = self.channelModel(x[:,:9])
        x2 = torch.cat([x1,x[:,9:]],dim=1)
        y = self.spaceModel(x2)
        return y

if __name__ == "__main__":
    spec_input = torch.rand(10, 18, 11, 11)
    yuv_input = torch.rand(10, 3, 11, 11)
    # print(dummy_input)
    model = TwoStreamMiddMaterial(18,3,3)
    predict = model(spec_input,yuv_input)
    print(predict.size())
