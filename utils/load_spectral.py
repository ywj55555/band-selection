import numpy as np
import warnings
warnings.filterwarnings("ignore")
import spectral.io.envi as envi
import time
import gc
from utils.os_helper import judegHdrDataType
# waveLength = [499.8, 514.8, 562.6, 583.8, 642.3, 702.9, 803.2, 678.0, 724.2, 837.8, 882.4]
# # band = [28, 35, 54, 61, 77, 90, 107, 85, 94, 112, 118]
# # band = [23, 49, 57, 63, 82, 90, 103, 109, 116]
# bandWaveLength = [450, 515, 560, 580, 640, 659, 680, 700, 725, 777, 800]
# band = [2,36,54,61,77,82,87,91,95,104,108]
# band = [x - 1 for x in band]
ALL_BAND_NUM = 128
# band = band - 1

def Sigmoid(x,gama = 0.1):
    x = (1/(1+(np.exp(-1 * x * gama))) - 0.5) * 2
    return x

def Tanh(x,gama = 0.1):
    x = np.tanh(x*gama)
    return x

def Exp(x,gama = 0.1):
    x = 1 - (np.exp(-1 * x * gama))
    return x

def envi_loader(dirpath, filename, bands, norma=True):
    # Test(/s)
    # load: 2.107914924621582
    # cut: 0.5144715309143066
    # normalize: 0.17873883247375488
    judegHdrDataType(dirpath, filename)
    enviData = envi.open(dirpath + filename + '.hdr', dirpath + filename + '.img')
    imgData = enviData.load()
    imgData = np.array(imgData, dtype=np.float32)
    # 选取9个波段
    # imgData = np.stack([imgData[:, :, band[0]], imgData[:, :, band[1]], imgData[:, :, band[2]], imgData[:, :, band[3]], \
    #                     imgData[:, :, band[4]], imgData[:, :, band[5]], imgData[:, :, band[6]], imgData[:, :, band[7]], \
    #                     imgData[:, :, band[8]]], axis=2)
    if len(bands) != ALL_BAND_NUM:
        # bandNums = len(bands)
        # imgData = np.stack([imgData[:, :, bands[x]] for x in range(bandNums)], axis=2)
        imgData = imgData[:, :, bands]
    # tt = time.time()
    if norma:
        imgData = envi_normalize(imgData)
    gc.collect()
    return imgData


def envi_normalize(imgData):
    img_max = np.max(imgData, axis=2 ,keepdims = True)
    return imgData / (img_max+0.0001)
    # return imgData / img_max.reshape(imgData.shape[0], imgData.shape[1], 1)


def transform(data, typeCode):
    row, col, channels = data.shape
    if typeCode == 0: # SKIN
        res = np.zeros((row, col, 8), dtype=np.float)
        res[:, :, 0] = (data[:, :, 6] - data[:, :, 4]) / (data[:, :, 6] + data[:, :, 4])
        res[:, :, 1] = data[:, :, 6]
        res[:, :, 2] = data[:, :, 0]
        res[:, :, 3] = data[:, :, 2] / data[:, :, 0]
        res[:, :, 4] = np.maximum(data[:, :, 4], data[:, :, 7]) - data[:, :, 8]
        res[:, :, 5] = data[:, :, 4] - data[:, :, 3]
        res[:, :, 6] = data[:, :, 3] - data[:, :, 0]
        res[:, :, 7] = data[:, :, 6] - data[:, :, 8]
        return res
    elif typeCode == 1: # CLOTH
        res = np.zeros((row, col, 6), dtype=np.float)
        res[:, :, 0] = (data[:, :, 6] - data[:, :, 4]) / (data[:, :, 6] + data[:, :, 4])
        res[:, :, 1] = data[:, :, 2] / data[:, :, 1]
        res[:, :, 2] = data[:, :, 5] / data[:, :, 4]
        res[:, :, 3] = data[:, :, 2] - data[:, :, 0]
        res[:, :, 4] = data[:, :, 4] - data[:, :, 2]
        res[:, :, 5] = data[:, :, 6]
        return res
    elif typeCode == 2: # PLANT
        res = np.zeros((row, col, 6), dtype=np.float)
        res[:, :, 0] = (data[:, :, 6] - data[:, :, 4]) / (data[:, :, 6] + data[:, :, 4])
        res[:, :, 1] = data[:, :, 7] / data[:, :, 2]
        res[:, :, 2] = data[:, :, 5] / data[:, :, 2]
        res[:, :, 3] = data[:, :, 2] / data[:, :, 1]
        res[:, :, 4] = data[:, :, 5] / data[:, :, 4]
        res[:, :, 5] = data[:, :, 4] / data[:, :, 2]
        return res
    else:
        return data


def transform2(data):
    row, col, channels = data.shape
    # skin
    res = np.zeros((row, col, 20), dtype=np.float)
    res[:, :, 0] = (data[:, :, 6] - data[:, :, 4]) / (data[:, :, 6] + data[:, :, 4])
    res[:, :, 1] = data[:, :, 6]
    res[:, :, 2] = data[:, :, 0]
    res[:, :, 3] = data[:, :, 2] / data[:, :, 0]
    res[:, :, 4] = np.maximum(data[:, :, 4], data[:, :, 7]) - data[:, :, 8]
    res[:, :, 5] = data[:, :, 4] - data[:, :, 3]
    res[:, :, 6] = data[:, :, 3] - data[:, :, 0]
    res[:, :, 7] = data[:, :, 6] - data[:, :, 8]
    # cloth
    res[:, :, 8] = (data[:, :, 6] - data[:, :, 4]) / (data[:, :, 6] + data[:, :, 4])
    res[:, :, 9] = data[:, :, 2] / data[:, :, 1]
    res[:, :, 10] = data[:, :, 5] / data[:, :, 4]
    res[:, :, 11] = data[:, :, 2] - data[:, :, 0]
    res[:, :, 12] = data[:, :, 4] - data[:, :, 2]
    res[:, :, 13] = data[:, :, 6]
    # PLANT
    res[:, :, 14] = (data[:, :, 6] - data[:, :, 4]) / (data[:, :, 6] + data[:, :, 4])
    res[:, :, 15] = data[:, :, 7] / data[:, :, 2]
    res[:, :, 16] = data[:, :, 5] / data[:, :, 2]
    res[:, :, 17] = data[:, :, 2] / data[:, :, 1]
    res[:, :, 18] = data[:, :, 5] / data[:, :, 4]
    res[:, :, 19] = data[:, :, 4] / data[:, :, 2]
    return res

def kindsOfFeatureTransformation(imgData, nora = True, eps = 0.001):
    # 传入的imgData 不要归一化
    # H W C(11)
    # row, col, channels = imgData.shape
    # bandWaveLength = [450, 515, 560, 580, 640, 659, 680, 700, 725, 777, 800]
    # band = [2, 36, 54, 61, 77, 82, 87, 91, 95, 104, 108] #原来128
    # order = [0, 1,  2   3    4   5   6   7   8    9   10] #抽取的11个波段
    featureList = []
    # ndwi
    greenBand = imgData[:,:,2]
    nir_band = imgData[:,:,10]
    # H W
    NDWI = (greenBand - nir_band) / (greenBand + nir_band + eps)
    featureList.append(NDWI)

    pillars_band1 = 7
    pillars_band2 = 9
    pillars_feature1 = imgData[:, :, pillars_band1]
    pillars_feature2 = imgData[:, :, pillars_band2]
    # pillars_feature = pillars_feature1 / (pillars_feature2 + eps)
    pillars_feature = (pillars_feature1 - pillars_feature2) / (pillars_feature1 + pillars_feature2 + eps)
    featureList.append(pillars_feature)

    bottle_band1 = 9
    bottle_band2 = 3
    bottle_feature1 = imgData[:, :, bottle_band1]
    bottle_feature2 = imgData[:, :, bottle_band2]
    # bottle_feature = bottle_feature1 / (bottle_feature2 + eps)
    bottle_feature = (bottle_feature1 - bottle_feature2) / (bottle_feature1 + bottle_feature2 + eps)
    featureList.append(bottle_feature)

    # draft
    draft_band1 = 5
    draft_band2 = 9
    # daytime
    draft_feature1 = imgData[:, :, draft_band1]
    draft_feature2 = imgData[:, :, draft_band2]
    # draft_cal_feature1 = draft_feature2 /(draft_feature1 + eps)
    draft_cal_feature1 = (draft_feature2 - draft_feature1) / (draft_feature1 + draft_feature2 + eps)
    featureList.append(draft_cal_feature1)

    # bandWaveLength = [450, 515, 560, 580, 640, 659, 680, 700, 725, 777, 800]
    # band = [2, 36, 54, 61, 77, 82, 87, 91, 95, 104, 108]
    # order =[0, 1,  2   3    4   5   6   7   8    9   10]
    # finger
    finger_band1 = 0
    finger_band2 = 3
    finger_band3 = 7
    finger_feature1 = imgData[:, :, finger_band1]
    finger_feature2 = imgData[:, :, finger_band2]
    finger_feature3 = imgData[:, :, finger_band3]
    # finger_cal_feature1 = finger_feature2 / (finger_feature1 + eps)
    # finger_cal_feature2 = finger_feature3 / (finger_feature2 + eps)
    finger_cal_feature1 = (finger_feature2 - finger_feature1) / (finger_feature2 + finger_feature1 + eps)
    finger_cal_feature2 = (finger_feature3 - finger_feature2) / (finger_feature3 + finger_feature2 + eps)
    featureList.append(finger_cal_feature1)
    featureList.append(finger_cal_feature2)

    # grass and tree
    plant_band1 = 5
    plant_band2 = 7
    plant_band3 = 7
    tree_band1 = 2
    tree_band2 = 3
    tree_band3 = 9
    tree_band4 = 10
    skin_band1 = 0
    black_tree_feature1 = imgData[:, :, tree_band3]
    black_tree_feature2 = imgData[:, :, tree_band4]
    # black_tree_feature = black_tree_feature2 / (black_tree_feature1 + eps)
    black_tree_feature = (black_tree_feature2 - black_tree_feature1) / (black_tree_feature2 + black_tree_feature1 + eps)
    featureList.append(black_tree_feature)

    plant_feature1 = imgData[:, :, plant_band1]
    plant_feature2 = imgData[:, :, plant_band2]
    plant_feature3 = imgData[:, :, plant_band3]
    tree_feature1 = imgData[:, :, tree_band1]
    tree_feature2 = imgData[:, :, tree_band2]

    # plant_feature = plant_feature2 / (plant_feature1 + eps)
    plant_feature = (plant_feature2 - plant_feature1) / (plant_feature2 + plant_feature1 + eps)
    # tree_feature = tree_feature2 / (tree_feature1 + eps)
    tree_feature = (tree_feature2 - tree_feature1) / (tree_feature2 + tree_feature1 + eps)
    # cloth_feature = plant_feature1
    # plant_feature9 = plant_feature1 / (plant_feature3 + eps)
    plant_feature9 = (plant_feature1 - plant_feature3) / (plant_feature1 + plant_feature3 + eps)

    featureList.append(plant_feature)
    featureList.append(tree_feature)
    # featureList.append(cloth_feature)
    featureList.append(plant_feature9)
    featureImgData = np.stack(featureList, axis = 2)
######全正值
    featureImgData = featureImgData / 2 + 0.5
######
    if nora:
        print("normalizing......")

        imgData = envi_normalize(imgData)
        # featureImgData = envi_normalize(featureImgData)

    fusionData = np.concatenate([imgData, featureImgData], axis = 2)

    return fusionData

# 三个 2500 随机 小模型 sig 和 tanh 波段平移以后的
def kindsOfFeatureTransformation_slope(imgData,  activate, nora = True, eps = 0.001):
    # 传入的imgData 不要归一化
    # H W C(11)
    # row, col, channels = imgData.shape
    # bandWaveLength = [450, 515, 560, 580, 640, 659, 680, 700, 725, 777, 800]
    # band = [2, 36, 54, 61, 77, 82, 87, 91, 95, 104, 108] #原来128
    # order = [0, 1,  2   3    4   5   6   7   8    9   10] #抽取的11个波段
    featureList = []
    # ndwi
    greenBand = imgData[:, :, 2]
    nir_band = imgData[:, :, 10]
    # H W
    NDWI = (greenBand - nir_band) / (greenBand + nir_band + eps)
    featureList.append(NDWI)
    pillars_band1 = 7
    pillars_band2 = 9
    pillars_feature1 = imgData[:, :, pillars_band1]
    pillars_feature2 = imgData[:, :, pillars_band2]
    pillars_feature = pillars_feature1 / (pillars_feature2 + eps)
    # if activate == ''
    pillars_feature = activate(pillars_feature)
    # pillars_feature = (pillars_feature1 - pillars_feature2) / (pillars_feature1 + pillars_feature2 + eps)
    featureList.append(pillars_feature)
    bottle_band1 = 9
    bottle_band2 = 3
    bottle_feature1 = imgData[:, :, bottle_band1]
    bottle_feature2 = imgData[:, :, bottle_band2]
    bottle_feature = bottle_feature1 / (bottle_feature2 + eps)
    bottle_feature = activate(bottle_feature)
    # bottle_feature = (bottle_feature1 - bottle_feature2) / (bottle_feature1 + bottle_feature2 + eps)
    featureList.append(bottle_feature)
    # draft
    draft_band1 = 5
    draft_band2 = 9
    # daytime
    draft_feature1 = imgData[:, :, draft_band1]
    draft_feature2 = imgData[:, :, draft_band2]
    draft_cal_feature1 = draft_feature2 / (draft_feature1 + eps)
    draft_cal_feature1 = activate(draft_cal_feature1)
    # draft_cal_feature1 = (draft_feature2 - draft_feature1) / (draft_feature1 + draft_feature2 + eps)
    featureList.append(draft_cal_feature1)

    # bandWaveLength = [450, 515, 560, 580, 640, 659, 680, 700, 725, 777, 800]
    # band = [2, 36, 54, 61, 77, 82, 87, 91, 95, 104, 108]
    # order =[0, 1,  2   3    4   5   6   7   8    9   10]
    # finger
    finger_band1 = 0
    finger_band2 = 3
    finger_band3 = 7
    finger_feature1 = imgData[:, :, finger_band1]
    finger_feature2 = imgData[:, :, finger_band2]
    finger_feature3 = imgData[:, :, finger_band3]
    finger_cal_feature1 = finger_feature2 / (finger_feature1 + eps)
    finger_cal_feature2 = finger_feature3 / (finger_feature2 + eps)
    finger_cal_feature1 = activate(finger_cal_feature1)
    finger_cal_feature2 = activate(finger_cal_feature2)
    # finger_cal_feature1 = (finger_feature2 - finger_feature1) / (finger_feature2 + finger_feature1 + eps)
    # finger_cal_feature2 = (finger_feature3 - finger_feature2) / (finger_feature3 + finger_feature2 + eps)
    featureList.append(finger_cal_feature1)
    featureList.append(finger_cal_feature2)

    # grass and tree
    plant_band1 = 5
    plant_band2 = 7
    plant_band3 = 7
    tree_band1 = 2
    tree_band2 = 3
    tree_band3 = 9
    tree_band4 = 10
    skin_band1 = 0
    black_tree_feature1 = imgData[:, :, tree_band3]
    black_tree_feature2 = imgData[:, :, tree_band4]
    black_tree_feature = black_tree_feature2 / (black_tree_feature1 + eps)
    black_tree_feature = activate(black_tree_feature)
    # black_tree_feature = (black_tree_feature2 - black_tree_feature1) / (black_tree_feature2 + black_tree_feature1 + eps)
    featureList.append(black_tree_feature)

    plant_feature1 = imgData[:, :, plant_band1]
    plant_feature2 = imgData[:, :, plant_band2]
    plant_feature3 = imgData[:, :, plant_band3]
    tree_feature1 = imgData[:, :, tree_band1]
    tree_feature2 = imgData[:, :, tree_band2]

    plant_feature = plant_feature2 / (plant_feature1 + eps)
    plant_feature = activate(plant_feature)
    # plant_feature = (plant_feature2 - plant_feature1) / (plant_feature2 + plant_feature1 + eps)
    tree_feature = tree_feature2 / (tree_feature1 + eps)
    tree_feature = activate(tree_feature)
    # tree_feature = (tree_feature2 - tree_feature1) / (tree_feature2 + tree_feature1 + eps)
    # cloth_feature = plant_feature1
    plant_feature9 = plant_feature1 / (plant_feature3 + eps)
    plant_feature9 = activate(plant_feature9)
    # plant_feature9 = (plant_feature1 - plant_feature3) / (plant_feature1 + plant_feature3 + eps)

    featureList.append(plant_feature)
    featureList.append(tree_feature)
    # featureList.append(cloth_feature)
    featureList.append(plant_feature9)
    featureImgData = np.stack(featureList, axis=2)
    ######全正值
    # featureImgData = featureImgData / 2 + 0.5
    ######
    if nora:
        print("normalizing......")

        imgData = envi_normalize(imgData)
        # featureImgData = envi_normalize(featureImgData)

    fusionData = np.concatenate([imgData, featureImgData], axis=2)

    return fusionData

if __name__ == '__main__':
    imgdata = np.random.randint(1,10,(100,100,11))
    imgdata = kindsOfFeatureTransformation(imgdata)
    print(imgdata.shape)
    # pass
'''
if __name__ == "__main__":
    envi_loader('E:/BUAA_Spetral_Data/pick/envi/', '20201031151257299')
'''
