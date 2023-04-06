import random
from skimage import io
import cv2 as cv
import cv2
import numpy as np
import os
from data.dictNew import *
from utils.load_spectral import *
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Variable
import math
import copy
import gc
import torch

CODE_STONE = 1
CODE_CHINA = 2
CODE_ROAD = 3
CODE_GREENLEAF = 4
CODE_OTHERPLANT = 5
CODE_SKIN = 6
CODE_NYLON = 7
CODE_COTTON = 8
CODE_CASHMERE = 9
CODE_SKY = 10
CODE_CAR = 11
CODE_IRON = 12
CODE_PLASTIC = 13
CODE_OTHER = 14

# code2label = [0, 0, 0, 0, 3, 3, 1, 2, 2, 2, 0, 0, 0, 0, 0]
code2label = [0,2,2,2,0,2,2,2,1,0,0,0,0,0,0,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0] #1:skin 2:cloth 3:plant 0:other
#0:其他 1：皮肤，2：衣物，3：植物

label2target = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

PLANT_SET = [CODE_GREENLEAF, CODE_OTHERPLANT]
CLOTH_SET = [CODE_NYLON, CODE_COTTON, CODE_CASHMERE]
SKIN_SET = [CODE_SKIN]

SKIN_TYPE = 0
CLOTH_TYPE = 1
PLANT_TYPE = 2

# DATA_TYPE = SKIN_TYPE
# DATA_TYPE = CLOTH_TYPE
DATA_TYPE = PLANT_TYPE

SELECT_ALL = 0  # select all pixels for training and testing
SELECT_RANDOM = 1  # select pixels randomly for training and testing

# env_data_dir = '/home/cjl/dataset/hyperSpectral/'
# label_data_dir = '/home/cjl/dataset/hyperSpectral/'
env_data_dir = '/home/cjl/dataset/envi/'
label_data_dir = '/home/cjl/dataset/label/'
# interval = 6
# max_nums = 15000
# Depth = 256
# label_data_dir_6se = '/home/cjl/data/sensor6/label/'
# tif_dir = '/home/cjl/data/sensor6/tif_data/'
#env_data_dir = 'E:/BUAA_Spetral_Data/hangzhou/envi/'
#label_data_dir = 'E:/BUAA_Spetral_Data/hangzhou/label/'

def convert_to_one_hot(y, C):
    return np.eye(C,dtype=np.int8)[y.reshape(-1)]

def generateData(dataType, num, length, typeCode,nora=True,selectMode=SELECT_RANDOM):
    Data = []
    Label = []
    # if dataType == 'train':
    #     dataFile = trainFile
    # elif dataType == 'test':
    #     dataFile = testFile
    if dataType == 'train':
        dataFile = trainFile
    elif dataType == 'test':
        dataFile = testFile
    elif dataType == 'all':
        dataFile = alltrainFile
    else:
        dataFile = testfile
    for file in dataFile:
        print(file)
        # t1 = time.time()
        imgLabel = io.imread(label_data_dir + file + '.png')
        # t2 = time.time()
        # imgData = envi_loader(env_data_dir, file,nora)
        imgData = envi_loader(os.path.join(env_data_dir, file[:8]) + '/', file, nora)
        # t3 = time.time()
        # imgData = transform2(imgData)
        # t4 = time.time()
        # print(imgData)
        # print(typeCode)
        pix, label = generateAroundDeltaPatchAllLen(imgData, imgLabel, typeCode, num, length, selectMode)
        #print(pix.shape)
        #print(label.shape)
        # t5 = time.time()
        # print('read label:', t2 - t1, ',read envi:', t3 - t2, ',transform:', t4 - t3, ',genrate:', t5 - t4)
        # read label: 3.676800489425659 ,read envi: 9.025498867034912 ,transform: 0.22420549392700195 ,genrate: 2.00569748878479
        Data.extend(pix)
        Label.extend(label)
    Data =np.array(Data,dtype=np.uint16)
    Label = np.array(Label)
    gc.collect()
    return Data, Label

def generateAroundDeltaPatchAllLen(imgData, imgLabel, typeCode, labelCount=2000, length=11, selectMode=SELECT_RANDOM):
    row, col, d = imgData.shape
    imgData = np.array(imgData)
    img = imgData.transpose(2, 0, 1)
    img = np.expand_dims(img, 0) #BCHW
    # img.shape -> (1, channels, row, col)
    pix = np.empty([0, d, length, length], dtype=np.uint16)
    labels = np.empty([0, 4])
    length1 = int(length / 2)
    for label in range(1, 31):  # 共有14类，取值为1、2、3、...、14
        label_index = np.where(imgLabel == label)  # 找到标签值等于label的下标点 size：2*个数
        pixels_nums = len(label_index[0])
        if pixels_nums == 0:
            continue
        delete_index = []
        for i in range(pixels_nums):
            if (label_index[0][i] <= length1 or label_index[0][i] >= row - length1 or label_index[1][i] <= length1 or
                    label_index[1][i] >= col - length1):
                delete_index.append(i)
        # 删除多列
        label_index = np.delete(label_index, delete_index, axis=1)
        pixels_nums = len(label_index[0])
        if pixels_nums == 0:
            continue

        tmp = np.array(label2target[code2label[label]])

        if selectMode == SELECT_ALL:
            for index in range(pixels_nums):
                labels = np.append(labels, np.reshape(tmp, [-1, 4]), axis=0)
                x = label_index[0][index]
                y = label_index[1][index]
                pixAround = img[:, :, x - length1: x + length1 + 1, y - length1: y + length1 + 1]  # 3*3*4
                pix = np.append(pix, pixAround, axis=0)
        elif selectMode == SELECT_RANDOM:
            for num in range(labelCount):
                index = random.randint(0, pixels_nums - 1)  # 前闭后闭区间
                # labels = np.append(labels, label)
                #print(np.reshape(tmp, [-1, 4]), np.reshape(tmp, [-1, 4]).shape)
                labels = np.append(labels, np.reshape(tmp, [-1, 4]), axis=0)
                x = label_index[0][index]
                y = label_index[1][index]
                pixAround = img[:, :, x - length1: x + length1 + 1, y - length1: y + length1 + 1]  # 3*3*4
                pix = np.append(pix, pixAround, axis=0)

    labels = np.array(labels).astype('int64')
    pix = np.array(pix, dtype=np.uint16)
    # print(pix.shape)
    # print('label shape:',labels.shape)
    return pix, labels



# def generateHyperData(data_type,length):
#     if data_type == 'meat':
#         dataFile = meatFile
#         class_nums = 5
#     elif data_type == 'fruit':
#         dataFile = fruitFile
#         class_nums = 6
#     elif data_type == 'oil':
#         dataFile = oilFile
#         class_nums = 3
#     else:
#         dataFile = testfile
#         class_nums = 5
#     trainData = np.empty([0, length, length, Depth], dtype=float)
#     trainLabel = np.empty([0, class_nums])
#     testData = np.empty([0, length, length, Depth], dtype=float)
#     testLabel = np.empty([0, class_nums])
#     length1 = int(length / 2)
#     for file in dataFile:
#         print(file)
#         # t1 = time.time()
#         imgLabel = io.imread(label_data_dir + file[:-4] + '.png')
#         # 稀疏采样 这样会类别不均 最好在 label循环里面进行 回到稀疏采样代码
#         # alldataNums = np.sum(imgLabel[length1:-length1, length1:-length1] !=0)
#         imgLabel[imgLabel == 0] = 255
#         imgLabel[imgLabel == class_nums] = 0 #转换背景
#         # print('the number of datapath: ',alldataNums)
#         enviData = envi.open(env_data_dir+file[:-4] + '.hdr', env_data_dir+file)
#         enviData = enviData.load()
#         enviData = np.array(enviData,dtype=float)
#         imgData = enviData.reshape(np.prod(enviData.shape[:2]), np.prod(enviData.shape[2:]))
#         imgData = preprocessing.scale(imgData)
#         traindata = imgData.reshape(enviData.shape[0], enviData.shape[1], enviData.shape[2])
#         traindata = np.expand_dims(traindata, 0) #B H W C
#         for label in range(class_nums):
#
#             tmp_label_index = np.where(imgLabel[length1:-length1, length1:-length1] == label)
#             tmp_class_cnts = len(tmp_label_index[0])
#             if tmp_class_cnts==0:
#                 continue
#             interval = 2
#             # 间隔1采样，数量除以4
#             if tmp_class_cnts > interval * interval * max_nums:
#                 mul = math.sqrt(tmp_class_cnts / max_nums)
#                 interval = round(mul)
#             # interval = 2 if tmp_class_cnts//max_nums<=2 else tmp_class_cnts//max_nums
#             cnt = math.ceil(imgLabel.shape[0] / interval)
#
#             imgLabel_tmp = copy.deepcopy(imgLabel)
#
#             mask = []
#             for i in range(cnt):
#                 mask += [mk for mk in range(i * interval,
#                                             interval * (i + 1) - 1 if interval * (i + 1) - 1 < imgLabel.shape[0] else
#                                             imgLabel.shape[0])]
#             imgLabel_tmp[mask, :] = 255
#             mask = []
#             cnt = math.ceil(imgLabel.shape[1] / interval)
#             # print('cnt: ', cnt)
#             for i in range(cnt):
#                 mask += [mk for mk in range(i * interval,
#                                             interval * (i + 1) - 1 if interval * (i + 1) - 1 < imgLabel.shape[1] else
#                                             imgLabel.shape[1])]
#             imgLabel_tmp[:, mask] = 255
#
#             label_index = np.where(imgLabel_tmp[length1:-length1, length1:-length1] == label)
#             new_count = len(label_index[0])
#             print(label,'sampling nums', new_count)
#             trainNums = math.floor(new_count*0.7)
#             # testNums = new_count-trainNums
#             allIndex = range(new_count)
#             trainIndex = random.sample(allIndex,trainNums)
#             testIndex = [ _ for _ in allIndex if _ not in trainIndex]
#             # tmp = np.array([0 for lab in range(class_nums) if lab!=label])
#             tmp = np.array(convert_to_one_hot(np.array([class_order for class_order in range(class_nums)]), class_nums)[
#                                label])
#             for index in trainIndex:
#                 x = label_index[0][index]
#                 y = label_index[1][index]
#                 pixAround = traindata[:,x:x + 2 * length1 + 1, y: y + 2 * length1 + 1, :]  # 21,21,bands
#                 trainData = np.append(trainData, pixAround, axis=0)
#                 # labels = np.append(labels, np.reshape(tmp, [-1, 4]), axis=0)
#                 trainLabel = np.append(trainLabel, np.reshape(tmp, [-1, class_nums]), axis=0)
#
#             for index in testIndex:
#                 x = label_index[0][index]
#                 y = label_index[1][index]
#                 pixAround = traindata[:,x: x + 2 * length1 + 1, y: y + 2 * length1 + 1, :]  # 21,21,bands
#                 testData = np.append(testData, pixAround, axis=0)
#                 # labels = np.append(labels, np.reshape(tmp, [-1, 4]), axis=0)
#                 testLabel = np.append(testLabel, np.reshape(tmp, [-1, class_nums]), axis=0)
#
#     return trainData, trainLabel,testData,testLabel