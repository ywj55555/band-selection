import numpy as np
import random
from skimage import io
import csv
# from utils.dictNew import *
from utils.bsUtils import JM_distance,B_distance,get_average_jm_score,cal_mean_spectral_divergence
from utils.load_spectral import envi_loader
from utils.os_helper import mkdir
import time
import os

# shanghai dataset
# envi_path = 'D:/raw_file/shanghai/'
# label_data_dir = 'D:/raw_file/shanghai/'
envi_path = '/home/cjl/dataset/envi/'
label_data_dir = '/home/cjl/dataset/label/'
allband = True
# band = [28, 35, 54, 61, 77, 90, 107, 85, 94]
band = [1, 35, 54, 61, 77, 90, 107, 85, 94] #手工选取 重新算一下得分
# band = [0, 26, 85, 77, 61, 70, 91, 12, 51] #SFS选取
if allband:
    spec_num = 128
else:
    spec_num = len(band)
#需要提前确定好类别！！
class_list = [i for i in range(1,30)] #30 is other
code2label = [0,2,2,2,0,2,2,2,1,0,0,0,0,0,0,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0]
cloth =[0,1,2,4,5,6]
other=[3]
other.extend([x for x in range(8,29)])
skin=7
# class_list = [1,3,4,5,8,10,12,14]

def getSingleCalcDistanceData(imgData,imgLabel,labelCount=300,class_list=None):
    '''
    :param imgData: 通道归一化后的光谱数据 已经经过波段选择了
    :param imgLabel: 原始31类标签图像
    :param labelCount: 每一类选取计算距离的数量,先保持所有类别数量相等，最后取数量最少的那类？？不太科学啊，还是每一类不固定数量了！！然后输入距离函数为均值和协方差矩阵
    :param class_list: 需要计算距离的类别 为列表形式
    :return: n,x,y n为类别数 b为样本数量 y为通道数量
    '''
    imgData = np.array(imgData) #h w c
    spec_num = imgData.shape[2]
    class_num = len(class_list)
    # pix = np.empty([class_num,0,spec_num], dtype=float)
    pix = [np.empty([0,spec_num], dtype=float) for _ in range(class_num)] #类别要注意对应
    for i,label in enumerate(class_list):
        label_index = np.where(imgLabel == label)  # 找到标签值等于label的下标点 size：2*个数
        pixels_nums = len(label_index[0])
        # print(pixels_nums)
        if pixels_nums <= labelCount:
            select_pix_list = range(pixels_nums)
            # tmplabelCount = pixels_nums
        #随机选 labelCount 个像素点计算距离
        # print(pixels_nums)
        else:
            select_pix_list = random.sample(range(pixels_nums),labelCount if labelCount<pixels_nums else pixels_nums)
        # print(labelCount) 这个为啥变0了
        # print(len(select_pix_list))
        # print(len(select_pix_list))
        for selpix in select_pix_list:
            # index = random.randint(0, pixels_nums - 1)  # 前闭后闭区间
            x = label_index[0][selpix]
            y = label_index[1][selpix]
            pixtmp = imgData[x,y].reshape(1,spec_num)
            pix[i] = np.append(pix[i], pixtmp, axis=0)
        # print(pix[i].shape)
    # for i in range(class_num):
    #     print(np.isnan(pix[i]).any())
    #     print(pix[i].shape)
    #     print(pix[i])

    return pix

def getAllCalcMSDScore(filelist,select_bands_list,labelCount=800,savepath="./",savenpy=True):
    class_num = len(class_list)
    # 每个元素 n,bands
    pixall = [np.empty([0, spec_num], dtype=float) for _ in range(class_num)]  # 类别要注意对应
    for file in filelist:
        # print(file)
        imgLabel = io.imread(label_data_dir + file + '.png')
        # imgData = envi_loader(envi_path, file,band,nora=False,allband=allband)
        imgData = envi_loader(os.path.join(envi_path,file[:8])+'/', file,band,nora=False,allband=allband) #归一化后的分布 因为后续也是基于归一化后数据进行分类
        pixtmp = getSingleCalcDistanceData(imgData,imgLabel,labelCount,class_list)
        singleLen = 0
        for class_order in range(class_num):
            pixall[class_order] = np.append(pixall[class_order],pixtmp[class_order], axis=0)#np.append 会重新开辟内存
            singleLen +=pixtmp[class_order].shape[0]
            # print(pixall[class_order].shape)
        print(file,"sampleing " ,singleLen)

    if savenpy:
        try:
            np.savez(savepath+'calc_msd_data.npz',*pixall)

        except Exception as e:
            print("error:", e)

    calc_msd_data = np.empty([0, spec_num], dtype=float)
    for i in range(class_num):
        calc_msd_data = np.append(calc_msd_data,pixall[i],axis=0)
    # print(calc_msd_data.shape)
    msd_score = []
    for select_bands in select_bands_list:
        # print(select_bands)
        tmp_calc_msd_data = calc_msd_data[:,select_bands]
        tmp_msd_score = cal_mean_spectral_divergence(tmp_calc_msd_data)
        msd_score.append(tmp_msd_score)
    return msd_score

if __name__ == "__main__":
    # 设置随机种子
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    log = './log/'
    npypath = './mean_cov_npy/'
    mkdir(log)
    # mkdir(npypath)
    csv2_save_path = log + 'MSD2.csv'
    pixall = np.load(npypath + 'calc_msd_data.npz')
    f2 = open(csv2_save_path, 'w', newline='')
    f2_csv = csv.writer(f2)
    # testlist = ['20210521095803026']
    select_bands_list = [[14, 28, 42, 56, 70, 84, 98, 112, 126],
                         [39, 10, 92, 88, 71, 121, 30, 77, 58],
                         [0, 26, 85, 77, 61, 70, 91, 12, 51],
                         [52, 77, 25, 68, 87, 92, 5, 37, 80],
                         [105, 72, 0, 22, 37, 29, 23, 8, 58],
                         [71, 118, 91, 78, 72, 77, 85, 121, 10],
                         [1, 35, 54, 61, 77, 90, 107, 85, 94]]
    time_start = time.time()
    # score = getAllCalcMSDScore(allFile,select_bands_list,800,npypath,True)
    # 就是全部数据合在一起，不分类别
    calc_msd_data = np.empty([0, spec_num], dtype=float)
    for tmp_pix in pixall.values():
        calc_msd_data = np.append(calc_msd_data, tmp_pix, axis=0)
    msd_score = []
    for select_bands in select_bands_list:
        # print(select_bands)
        tmp_calc_msd_data = calc_msd_data[:, select_bands]
        tmp_msd_score = cal_mean_spectral_divergence(tmp_calc_msd_data)
        msd_score.append(tmp_msd_score)
    # score = cal_mean_spectral_divergence(allband_sample)
    time_end = time.time()
    print('cost: ',time_end-time_start,' s')
    # print(jm)
    # print(np.isnan(jm))
    f2_csv.writerow([" ","equally_spaced","random_select","SFS","SBS","EAS","BS-Nets","hand_select"])
    finalScore = ['final MSD-Score: ']
    finalScore.extend(msd_score)
    f2_csv.writerow(finalScore)
    print(msd_score)
    f2.close()