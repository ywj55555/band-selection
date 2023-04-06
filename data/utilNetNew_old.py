import random
from skimage import io
import cv2 as cv
import numpy as np
from data.dictNew import *
from utils.load_spectral import *
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from torch.autograd import Variable
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

code2label = [0, 0, 0, 0, 3, 3, 1, 2, 2, 2, 0, 0, 0, 0, 0] #3植物 2衣物 1皮肤 0其他
label2target = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] #onehot编码

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

env_data_dir = '/data3/chenjialin/hangzhou_data/envi/'
label_data_dir = '/data3/chenjialin/hangzhou_data/label/'
#env_data_dir = 'E:/BUAA_Spetral_Data/hangzhou/envi/'
#label_data_dir = 'E:/BUAA_Spetral_Data/hangzhou/label/'


def generateData(dataType, num, length, typeCode, selectMode=SELECT_RANDOM):
    Data = []
    Label = []
    if dataType == 'train':
        dataFile = trainFile
    elif dataType == 'test':
        dataFile = testFile
    for file in dataFile:
        print(file)
        # t1 = time.time()
        imgLabel = io.imread(label_data_dir + file + '.png')
        # t2 = time.time()
        # imgData , imgData_cut = envi_loader_cut(env_data_dir, file)
        imgData = envi_loader(env_data_dir, file)
        # t3 = time.time()
        imgData = transform2(imgData)  #
        # imgData = transform_cut(imgData,imgData_cut) #特征设计，返回设计的特征且带有原始数据的切面归一化
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
    return Data, Label


def generateAroundDeltaPatchAllLen(imgData, imgLabel, typeCode, labelCount=2000, length=11, selectMode=SELECT_RANDOM):
    row, col, d = imgData.shape
    imgData = np.array(imgData)
    img = imgData.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    # img.shape -> (1, channels, row, col)
    pix = np.empty([0, d, length, length], dtype=float)
    labels = np.empty([0, 4])
    length1 = int(length / 2)
    for label in range(1, 15):  # 共有14类，取值为1、2、3、...、14
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
    pix = np.array(pix, dtype=float)
    # print(pix.shape)
    # print('label shape:',labels.shape)
    return pix, labels
