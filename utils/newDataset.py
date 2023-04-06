import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from data.utilNetNew import *
import os

# env_data_dir = '/data3/chenjialin/hangzhou_data/envi/'
# env_sh_data_dir = '/data3/chenjialin/hangzhou_data/envi_shanghai/'
# label_data_dir = '/data3/chenjialin/hangzhou_data/label/'
label_dict = {'__skin':1,'_cloth':2,'_plant':3,'_other':0}
label2target = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
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

# 问问高斯核怎么弄
# class MyDataset(torch.utils.data.Dataset):
#     # file_list为所有npy文件列表
#     def __init__(self, file_list,data_path):
#         self.Data = file_list
#         self.data_path = data_path
#     def __getitem__(self, index):
#         # if os.path.exists(self.data_path+self.Data[index]+'.npy'):
#         #     # print(self.Data[index])
#         #     img = np.load(self.data_path+self.Data[index]+'.npy')
#         #     label = np.load(self.data_path+self.Data[index]+'_label.npy')
#         # else:
#         #     # print(self.Data[index])
#         #     img,label=generateDataset(self.Data[index])
#         #     # print(self.Data[index])
#         #     img = np.array(img)
#         #     label = np.array(label)
#         #     np.save(self.data_path+self.Data[index]+'.npy',img)
#         #     np.save(self.data_path + self.Data[index] + '_label.npy', label)
#         #     gc.collect()
#         img = np.load(self.data_path + self.Data[index])
#         label = label2target[label_dict[self.Data[index][-13:-7]]]
#         label = np.array(label)
#         # img = self.Data[index]
#         # label = self.Label[index]
#         return img, label
#
#     def __len__(self):
#         return len(self.Data)

#
# class MaterialSubModel(nn.Module):
#     def __init__(self, in_channels=20, out_channels=4, kernel_size = 11,padding_mode='reflect',mid_channels_1=16, mid_channels_2=32, mid_channels_3=8):
#         super(MaterialSubModel, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels_1, kernel_size=kernel_size, stride=1, padding=kernel_size//2,padding_mode=padding_mode),
#             nn.LeakyReLU()
#             #nn.ReLU()
#             #nn.ReLU(inplace=True)
#         )
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(mid_channels_1, mid_channels_2, kernel_size=kernel_size, stride=1, padding=kernel_size//2,padding_mode=padding_mode),
#             nn.LeakyReLU()
#             # nn.ReLU()
#         )
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(mid_channels_2, mid_channels_3, kernel_size=kernel_size, stride=1, padding=kernel_size//2,padding_mode=padding_mode),
#             nn.LeakyReLU()
#             # nn.ReLU()
#         )
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(mid_channels_3, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2,padding_mode=padding_mode),
#             nn.LeakyReLU()
#             # nn.ReLU()
#         )
#
#     def forward(self, x):
#         x=self.layer1(x)
#         #print('layer1:',x.size())
#         x=self.layer2(x)
#         #print('layer2:', x.size())
#         x = self.layer3(x)
#         #print('layer3:', x.size())
#         x = self.layer4(x)
#         #x = x.view(x.size(0),-1)
# #        print('layer4:', x.size())
#         #x=self.linear(x)
#         return x
#
#
# class MaterialModel(nn.Module):
#     def __init__(self):
#         super(MaterialModel, self).__init__()
#         self.subModel_skin = MaterialSubModel(8, 2)
#         self.subModel_cloth = MaterialSubModel(6, 2)
#         self.subModel_plant = MaterialSubModel(6, 2)
#         self.linear = nn.Linear(6, 4)
#
#     def forward(self, x):
#         x1 = x[:, 0:8, :, :]
#         x1 = self.subModel_skin(x1)
#         #print(x1.size())
#         x2 = x[:, 8:14, :, :]
#         x2 = self.subModel_cloth(x2)
#         #print(x2.size())
#         x3 = x[:, 14:20, :, :]
#         x3 = self.subModel_plant(x3)
#         #print(x3.size())
#         x = torch.cat([x1, x2, x3], 1)
#         x = x.squeeze()
#         x = x.squeeze()
#         #print(x.size())
#         x = self.linear(x)
#         return x
#
# if __name__ == "__main__":
#     dummy_input = torch.rand(10, 20, 1288, 1233)
#     print(dummy_input.size())
#     model = MaterialSubModel()
#     predict = model(dummy_input)
#     print(predict.size())
