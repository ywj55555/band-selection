import sys
sys.path.append('../')
import torch.nn as nn
import torch
from model.activation import mish
from model.attontion import CAM_Module,PAM_Module
import math
class DBDA_network_MISH_full_conv(nn.Module):
    def __init__(self, band, classes,channnel1=24):
        super(DBDA_network_MISH_full_conv, self).__init__()

        # spectral branch
        self.name = 'DBDA_network_MISH_full_conv'
        # 输入 9*11*11
        self.conv11 = nn.Conv3d(in_channels=1, out_channels=channnel1,
                                kernel_size=(3, 1, 1), stride=(1, 1, 1),padding=(1,0,0))
        # Dense block
        self.batch_norm11 = nn.Sequential(
                                    nn.BatchNorm3d(channnel1,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv12 = nn.Conv3d(in_channels=channnel1, out_channels=channnel1//2, padding=(1,0, 0),
                                kernel_size=(3, 1, 1), stride=(1, 1, 1))
        self.batch_norm12 = nn.Sequential(
                                    nn.BatchNorm3d(channnel1 + channnel1//2, eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv13 = nn.Conv3d(in_channels=channnel1 + channnel1//2, out_channels=channnel1//2,padding=(1,0, 0),
                                kernel_size=(3, 1, 1), stride=(1, 1, 1))
        self.batch_norm13 = nn.Sequential(
                                    nn.BatchNorm3d(channnel1+2*(channnel1//2), eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv14 = nn.Conv3d(in_channels=channnel1+2*(channnel1//2), out_channels=channnel1//2, padding=(1,0, 0),
                                kernel_size=(3, 1, 1), stride=(1, 1, 1))
        self.batch_norm14 = nn.Sequential(
                                    nn.BatchNorm3d(channnel1 + 3*(channnel1//2), eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        # kernel_3d = math.floor((band - 6) / 2) 压缩成depth = 1
        kernel_3d = math.ceil((band + 2 - 3) / 1 + 1)
        self.conv15 = nn.Conv3d(in_channels=channnel1 + 3*(channnel1//2), out_channels=channnel1 + 3*(channnel1//2),
                                kernel_size=(kernel_3d,1, 1), stride=(1, 1, 1)) # kernel size随数据变化


        # Spatial Branch
        self.conv21 = nn.Conv3d(in_channels=1, out_channels=channnel1,
                                kernel_size=(band,1, 1), stride=(1, 1, 1))
        # Dense block
        self.batch_norm21 = nn.Sequential(
                                    nn.BatchNorm3d(channnel1, eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv22 = nn.Conv3d(in_channels=channnel1, out_channels=channnel1//2, padding=(0 ,1, 1),
                                kernel_size=(1, 3, 3), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(
                                    nn.BatchNorm3d(channnel1 + channnel1//2, eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv23 = nn.Conv3d(in_channels=channnel1 + channnel1//2, out_channels=channnel1//2, padding=(0 ,1, 1),
                                kernel_size=(1, 3, 3), stride=(1, 1, 1))
        self.batch_norm23 = nn.Sequential(
                                    nn.BatchNorm3d(channnel1 + 2*(channnel1//2), eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv24 = nn.Conv3d(in_channels=channnel1 + 2*(channnel1//2), out_channels=channnel1//2, padding=(0 ,1, 1),
                                kernel_size=(1, 3, 3), stride=(1, 1, 1))

        #这个干嘛用的？
        # self.conv25 = nn.Sequential(
        #                         nn.Conv3d(in_channels=1, out_channels=1, padding=(1, 1, 0),
        #                         kernel_size=(3, 3, 2), stride=(1, 1, 1)),
        #                         nn.Sigmoid()
        # )

        self.batch_norm_spectral = nn.Sequential(
                                    nn.BatchNorm3d(channnel1+3*(channnel1//2),  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new(),
                                    #swish(),
            mish(),
                                    nn.Dropout(p=0.5)
        )
        self.batch_norm_spatial = nn.Sequential(
                                    nn.BatchNorm3d(channnel1 + 3*(channnel1//2),  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new(),
                                    #swish(),
            mish(),
                                    nn.Dropout(p=0.5)
        )

        #固定尺寸
        # self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.global_pooling = nn.AvgPool3d(kernel_size=(1, 11, 11), stride=(1, 1, 1))
        #改为全卷积层
        # self.full_connection = nn.Sequential(
        #                         #nn.Dropout(p=0.5),
        #                         nn.Linear(120, classes) # ,
        #                         # nn.Softmax()
        # )
        self.full_conv = nn.Sequential(
            # # nn.Dropout(p=0.5),
            # nn.Linear(120, classes)  # ,
            nn.Conv2d(in_channels=(channnel1 + 3*(channnel1//2))*2, out_channels=classes, kernel_size=(1, 1), stride=(1, 1))
            # # nn.Softmax()
        )

        self.attention_spectral = CAM_Module(channnel1 + 3*(channnel1//2))
        self.attention_spatial = PAM_Module(channnel1 + 3*(channnel1//2))

        #fc = Dense(classes, activation='softmax', name='output1',
        #           kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    def forward(self, X):
        # spectral
        x11 = self.conv11(X)
        #print('x11', x11.shape)
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)
        #print('x12', x12.shape)

        x13 = torch.cat((x11, x12), dim=1)
        #print('x13', x13.shape)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)
        #print('x13', x13.shape)

        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        # print('x15', x15.shape)

        x16 = self.batch_norm14(x15)
        x16 = self.conv15(x16)
        #print('x16', x16.shape)  # 7*7*97, 60

        #print('x16', x16.shape)
        # 光谱注意力通道
        x1 = self.attention_spectral(x16)
        x1 = torch.mul(x1, x16)


        # spatial
        #print('x', X.shape)
        x21 = self.conv21(X)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x21, x22, x23, x24), dim=1)
        #print('x25', x25.shape)
        # x25 = x25.permute(0, 4, 2, 3, 1)
        #print('x25', x25.shape)

        # 空间注意力机制
        #小块训练先去掉空间注意力，整图训练再加上
        # x2 = self.attention_spatial(x25)
        # x2 = torch.mul(x2, x25)
        x2 = x25

        # model1
        x1 = self.batch_norm_spectral(x1)
        x1 = self.global_pooling(x1)
        # x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        x1 = x1.squeeze(-3)
        x2 = self.batch_norm_spatial(x2)
        x2= self.global_pooling(x2)
        # x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)
        x2 = x2.squeeze(-3)

        x_pre = torch.cat((x1, x2), dim=1)
        #print('x_pre', x_pre.shape)

        # model2
        # x1 = torch.mul(x2, x16)
        # x2 = torch.mul(x2, x25)
        # x_pre = x1 + x2
        #
        #
        # x_pre = x_pre.view(x_pre.shape[0], -1)
        output = self.full_conv(x_pre)
        # output = self.fc(x_pre)
        return output

class DBDA_network_MISH_without_bn(nn.Module):
    def __init__(self, band, classes,channnel1=24):
        super(DBDA_network_MISH_without_bn, self).__init__()

        # spectral branch
        self.name = 'DBDA_MISH'
        # 输入 9*11*11
        self.conv11 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=channnel1,
                                kernel_size=(5, 1, 1), stride=(1, 1, 1),padding=(2,0,0)),
                                    mish())
        # Dense block
        # self.batch_norm11 = nn.Sequential(
        #                             nn.BatchNorm3d(channnel1,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
        #                             #gelu_new()
        #                             #swish()
        #     mish()
        # )
        self.conv12 =nn.Sequential( nn.Conv3d(in_channels=channnel1, out_channels=channnel1//2, padding=(2,0, 0),
                                kernel_size=(5, 1, 1), stride=(1, 1, 1)),mish())
        # self.batch_norm12 = nn.Sequential(
        #                             nn.BatchNorm3d(channnel1 + channnel1//2, eps=0.001, momentum=0.1, affine=True),
        #                             #gelu_new()
        #                             #swish()
        #     mish()
        # )
        self.conv13 =nn.Sequential( nn.Conv3d(in_channels=channnel1 + channnel1//2, out_channels=channnel1//2,padding=(2,0, 0),
                                kernel_size=(5, 1, 1), stride=(1, 1, 1)), mish())
        # self.batch_norm13 = nn.Sequential(
        #                             nn.BatchNorm3d(channnel1+2*(channnel1//2), eps=0.001, momentum=0.1, affine=True),
        #                             #gelu_new()
        #                             #swish()
        #     mish()
        # )
        self.conv14 =nn.Sequential( nn.Conv3d(in_channels=channnel1+2*(channnel1//2), out_channels=channnel1//2, padding=(2,0, 0),
                                kernel_size=(5, 1, 1), stride=(1, 1, 1)), mish())
        # self.batch_norm14 = nn.Sequential(
        #                             nn.BatchNorm3d(channnel1 + 3*(channnel1//2), eps=0.001, momentum=0.1, affine=True),
        #                             #gelu_new()
        #                             #swish()
        #     mish()
        # )
        # kernel_3d = math.floor((band - 6) / 2) 压缩成depth = 1
        kernel_3d = math.floor((band + 4 - 5) / 1 + 1)
        self.conv15 = nn.Conv3d(in_channels=channnel1 + 3*(channnel1//2), out_channels=channnel1 + 3*(channnel1//2),
                                kernel_size=(kernel_3d,1, 1), stride=(1, 1, 1)) # kernel size随数据变化


        # Spatial Branch
        self.conv21 =nn.Sequential( nn.Conv3d(in_channels=1, out_channels=channnel1,
                                kernel_size=(band,1, 1), stride=(1, 1, 1)),mish())
        # Dense block
        # self.batch_norm21 = nn.Sequential(
        #                             nn.BatchNorm3d(channnel1, eps=0.001, momentum=0.1, affine=True),
        #                             #gelu_new()
        #                             #swish()
        #     mish()
        # )
        self.conv22 =nn.Sequential( nn.Conv3d(in_channels=channnel1, out_channels=channnel1//2, padding=(0 ,1, 1),
                                kernel_size=(1, 3, 3), stride=(1, 1, 1)),mish())
        # self.batch_norm22 = nn.Sequential(
        #                             nn.BatchNorm3d(channnel1 + channnel1//2, eps=0.001, momentum=0.1, affine=True),
        #                             #gelu_new()
        #                             #swish()
        #     mish()
        # )
        self.conv23 =nn.Sequential( nn.Conv3d(in_channels=channnel1 + channnel1//2, out_channels=channnel1//2, padding=(0 ,1, 1),
                                kernel_size=(1, 3, 3), stride=(1, 1, 1)),mish())
        # self.batch_norm23 = nn.Sequential(
        #                             nn.BatchNorm3d(channnel1 + 2*(channnel1//2), eps=0.001, momentum=0.1, affine=True),
        #                             #gelu_new()
        #                             #swish()
        #     mish()
        # )
        self.conv24 =nn.Sequential( nn.Conv3d(in_channels=channnel1 + 2*(channnel1//2), out_channels=channnel1//2, padding=(0 ,1, 1),
                                kernel_size=(1, 3, 3), stride=(1, 1, 1)), mish())

        #这个干嘛用的？
        # self.conv25 = nn.Sequential(
        #                         nn.Conv3d(in_channels=1, out_channels=1, padding=(1, 1, 0),
        #                         kernel_size=(3, 3, 2), stride=(1, 1, 1)),
        #                         nn.Sigmoid()
        # )

        # self.batch_norm_spectral = nn.Sequential(
        #                             nn.BatchNorm3d(channnel1+3*(channnel1//2),  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
        #                             #gelu_new(),
        #                             #swish(),
        #     mish(),
        #                             nn.Dropout(p=0.5)
        # )
        # self.batch_norm_spatial = nn.Sequential(
        #                             nn.BatchNorm3d(channnel1 + 3*(channnel1//2),  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
        #                             #gelu_new(),
        #                             #swish(),
        #     mish(),
        #                             nn.Dropout(p=0.5)
        # )

        #固定尺寸
        # self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.global_pooling = nn.AvgPool3d(kernel_size=(1, 11, 11), stride=(1, 1, 1))
        #改为全卷积层
        # self.full_connection = nn.Sequential(
        #                         #nn.Dropout(p=0.5),
        #                         nn.Linear(120, classes) # ,
        #                         # nn.Softmax()
        # )
        self.full_conv = nn.Sequential(
            # # nn.Dropout(p=0.5),
            # nn.Linear(120, classes)  # ,
            nn.Conv2d(in_channels=(channnel1 + 3*(channnel1//2))*2, out_channels=classes, kernel_size=(1, 1), stride=(1, 1))
            # # nn.Softmax()
        )

        self.attention_spectral = CAM_Module(channnel1 + 3*(channnel1//2))
        self.attention_spatial = PAM_Module(channnel1 + 3*(channnel1//2))

        #fc = Dense(classes, activation='softmax', name='output1',
        #           kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    def forward(self, X):
        # spectral
        x11 = self.conv11(X)
        #print('x11', x11.shape)
        # x12 = self.batch_norm11(x11)
        x12 = self.conv12(x11)
        #print('x12', x12.shape)

        x13 = torch.cat((x11, x12), dim=1)
        #print('x13', x13.shape)
        # x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)
        #print('x13', x13.shape)

        x14 = torch.cat((x11, x12, x13), dim=1)
        # x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        # print('x15', x15.shape)

        # x16 = self.batch_norm14(x15)
        x16 = self.conv15(x15) #没有激活
        #print('x16', x16.shape)  # 7*7*97, 60

        #print('x16', x16.shape)
        # 光谱注意力通道，长距离构建关系，有点像transformer
        x1 = self.attention_spectral(x16)
        x1 = torch.mul(x1, x16)


        # spatial
        #print('x', X.shape)
        x21 = self.conv21(X)
        # x22 = self.batch_norm21(x21)
        x22 = self.conv22(x21)

        x23 = torch.cat((x21, x22), dim=1)
        # x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        # x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x21, x22, x23, x24), dim=1)
        #print('x25', x25.shape)
        # x25 = x25.permute(0, 4, 2, 3, 1)
        #print('x25', x25.shape)

        # 空间注意力机制
        #小块训练先去掉空间注意力，整图训练再加上
        # x2 = self.attention_spatial(x25)
        # x2 = torch.mul(x2, x25)
        x2 = x25

        # model1
        # x1 = self.batch_norm_spectral(x1)
        x1 = self.global_pooling(x1)
        # x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        x1 = x1.squeeze(-3)
        # x2 = self.batch_norm_spatial(x2)
        x2= self.global_pooling(x2)
        # x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)
        x2 = x2.squeeze(-3)

        x_pre = torch.cat((x1, x2), dim=1)
        #print('x_pre', x_pre.shape)

        # model2
        # x1 = torch.mul(x2, x16)
        # x2 = torch.mul(x2, x25)
        # x_pre = x1 + x2
        #
        #
        # x_pre = x_pre.view(x_pre.shape[0], -1)
        output = self.full_conv(x_pre)
        # output = self.fc(x_pre)
        return output

class DBDA_network_MISH(nn.Module):
    def __init__(self, band, classes):
        super(DBDA_network_MISH, self).__init__()

        # spectral branch
        self.name = 'DBDA_MISH'
        self.conv11 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # Dense block
        self.batch_norm11 = nn.Sequential(
                                    nn.BatchNorm3d(24,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv12 = nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm12 = nn.Sequential(
                                    nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv13 = nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm13 = nn.Sequential(
                                    nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv14 = nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm14 = nn.Sequential(
                                    nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        # kernel_3d = math.floor((band - 6) / 2) #应该是ceil !! 看一下输出尺寸就好了
        kernel_3d = math.ceil((band - 6) / 2)  #
        self.conv15 = nn.Conv3d(in_channels=60, out_channels=60,
                                kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1)) # kernel size随数据变化


        # Spatial Branch
        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, band), stride=(1, 1, 1))
        # Dense block
        self.batch_norm21 = nn.Sequential(
                                    nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv22 = nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(
                                    nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv23 = nn.Conv3d(in_channels=36, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm23 = nn.Sequential(
                                    nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv24 = nn.Conv3d(in_channels=48, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))


        self.conv25 = nn.Sequential(
                                nn.Conv3d(in_channels=1, out_channels=1, padding=(1, 1, 0),
                                kernel_size=(3, 3, 2), stride=(1, 1, 1)),
                                nn.Sigmoid()
        )

        self.batch_norm_spectral = nn.Sequential(
                                    nn.BatchNorm3d(60,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new(),
                                    #swish(),
            mish(),
                                    nn.Dropout(p=0.5)
        )
        self.batch_norm_spatial = nn.Sequential(
                                    nn.BatchNorm3d(60,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new(),
                                    #swish(),
            mish(),
                                    nn.Dropout(p=0.5)
        )


        self.global_pooling = nn.AdaptiveAvgPool3d(1) #因为这一行代码 把前面 kernel_3d = math.floor((band - 6) / 2) 的错误弥补了，但是意义不对
        self.full_connection = nn.Sequential(
                                #nn.Dropout(p=0.5),
                                nn.Linear(120, classes) # ,
                                # nn.Softmax()
        )

        self.attention_spectral = CAM_Module(60)
        self.attention_spatial = PAM_Module(60)

        #fc = Dense(classes, activation='softmax', name='output1',
        #           kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    def forward(self, X):
        # spectral
        x11 = self.conv11(X)
        #print('x11', x11.shape)
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)
        #print('x12', x12.shape)

        x13 = torch.cat((x11, x12), dim=1)
        #print('x13', x13.shape)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)
        #print('x13', x13.shape)

        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        # print('x15', x15.shape)

        x16 = self.batch_norm14(x15)
        x16 = self.conv15(x16)
        #print('x16', x16.shape)  # 7*7*97, 60

        #print('x16', x16.shape)
        # 光谱注意力通道
        x1 = self.attention_spectral(x16)
        x1 = torch.mul(x1, x16)


        # spatial
        #print('x', X.shape)
        x21 = self.conv21(X)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x21, x22, x23, x24), dim=1)
        #print('x25', x25.shape)
        # x25 = x25.permute(0, 4, 2, 3, 1)
        #print('x25', x25.shape)

        # 空间注意力机制
        x2 = self.attention_spatial(x25)
        x2 = torch.mul(x2, x25)

        # model1
        x1 = self.batch_norm_spectral(x1)
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        x2 = self.batch_norm_spatial(x2)
        x2= self.global_pooling(x2)
        x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)

        x_pre = torch.cat((x1, x2), dim=1)
        #print('x_pre', x_pre.shape)

        # model2
        # x1 = torch.mul(x2, x16)
        # x2 = torch.mul(x2, x25)
        # x_pre = x1 + x2
        #
        #
        # x_pre = x_pre.view(x_pre.shape[0], -1)
        output = self.full_connection(x_pre)
        # output = self.fc(x_pre)
        return output


if __name__=='__main__':
    model = DBDA_network_MISH_full_conv(11, 2, 4).cuda()
    # img = torch.rand((2,9,11,11)).cuda()
    # img = img.reshape(2,1,9,11,11)
    # torch.rand(1, 128, 1415, 1859).cuda()
    img2 = torch.rand((1, 11,1415, 1859)).cuda()
    img2 = img2.unsqueeze(1) #增加通道位置
    # output = model(img)
    with torch.no_grad():
        output2 = model(img2)
        # print(output.shape)
        print(output2.shape)