import numpy as np
import random
from skimage import io
import os
import csv
from utils.bsUtils import JM_distance,B_distance,get_average_jm_score
from utils.dictNew import *
from utils.load_spectral import envi_loader
from utils.os_helper import mkdir
import time

# shanghai dataset
envi_path = 'D:/raw_file/shanghai/'
label_data_dir = 'D:/raw_file/shanghai/'
# envi_path = '/home/cjl/dataset/envi/'
# label_data_dir = '/home/cjl/dataset/label/'
allband = False
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
        if pixels_nums <= 150:
            continue
        #随机选 labelCount 个像素点计算距离
        # print(pixels_nums)
        select_pix_list = random.sample(range(pixels_nums),labelCount if labelCount<pixels_nums else pixels_nums)
        # print(len(select_pix_list))
        for selpix in select_pix_list:
            # index = random.randint(0, pixels_nums - 1)  # 前闭后闭区间
            x = label_index[0][selpix]
            y = label_index[1][selpix]
            pixtmp = imgData[x,y].reshape(1,spec_num)
            pix[i] = np.append(pix[i], pixtmp, axis=0)
    # for i in range(class_num):
    #     print(np.isnan(pix[i]).any())
    #     print(pix[i].shape)
    #     print(pix[i])

    return pix

def getAllCalcDistanceData(filelist,labelCount=300,disType="JM",savepath="./",savenpy=True):
    class_num = len(class_list)
    pixall = [np.empty([0, spec_num], dtype=float) for _ in range(class_num)]  # 类别要注意对应
    for file in filelist:
        print(file)
        imgLabel = io.imread(label_data_dir + file + '.png')
        imgData = envi_loader(envi_path, file,band,nora=False,allband=allband)
        # imgData = envi_loader(os.path.join(envi_path,file[:8])+'/', file,band,nora=False,allband=allband) #归一化后的分布 因为后续也是基于归一化后数据进行分类
        pixtmp = getSingleCalcDistanceData(imgData,imgLabel,labelCount,class_list)
        for class_order in range(class_num):
            pixall[class_order] = np.append(pixall[class_order],pixtmp[class_order], axis=0)#np.append 会重新开辟内存
    # for i in range(class_num):
    #     print(np.isnan(pixall[i]).any())
    #     print(pixall[i].shape)
    #     print(pixall[i])
    mean_v = [np.nanmean(specdata,axis=0) for specdata in pixall]
    mean_v = np.array(mean_v) #shape: n,spec_num
    # print(mean_v)
    # for i in range(mean_v.shape[0]):
    #     print(np.isnan(mean_v[i]).any())
    #     print(mean_v[i].shape)
    #     print(mean_v[i])
    cov_v = [np.cov(specdata.transpose(1,0),ddof=1) for specdata in pixall] #shape: n,spec_num,spec_num
    cov_v = np.array(cov_v)
    # for i in range(cov_v.shape[0]):
    #     if np.isnan(cov_v[i]).any():
    #         continue
    #     print(np.isnan(cov_v[i]).any())
    #     print(cov_v[i].shape)
    #     print(cov_v[i])
    # print(cov_v)
    if savenpy:
        # 离线保存用于计算JM距离，加快搜索速度
        np.save(savepath+'mean.npy',mean_v)
        np.save(savepath + 'cov.npy', cov_v)
    if disType=="JM":
        JM_dist = JM_distance(mean_v,cov_v)
        return JM_dist
    elif disType=="B":
        B_dist = B_distance(mean_v, cov_v)
        return B_dist
    else:
        return None

if __name__ == "__main__":
    # 设置随机种子
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    log = './log/'
    npypath = './mean_cov_npy/'
    mkdir(log)
    # mkdir(npypath)
    csv2_save_path = log + 'JM_distance_hand_Selected_9band3.csv'
    f2 = open(csv2_save_path, 'w', newline='')
    f2_csv = csv.writer(f2)
    f2_csv.writerow([" "] + class_list)
    # testlist = ['20210521095803026']
    time_start = time.time()
    mean_v = np.load(npypath+'mean.npy')
    mean_v = mean_v[:,band]
    print(mean_v.shape)
    cov_v = np.load(npypath+'cov.npy')
    print(cov_v.shape)
    # 128 波段时，协方差矩阵行列式为0了？？ 9波段反而不为0？？
    # 128个波段之间存在线性相关的波段，随机抽取的9个就线性无关，所以，后续波段选取计算JM距离时候也需要考虑这个问题
    cov_v = cov_v[:,band]
    cov_v = cov_v[:,:, band]
    print(cov_v.shape)
    jm = JM_distance(mean_v, cov_v)
    # jm = getAllCalcDistanceData(allFile,800,savepath=npypath)
    # code2label = [0, 2, 2, 2, 0, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    score = get_average_jm_score(mean_v,cov_v,cloth=cloth,other=other,skin=skin)
    time_end = time.time()
    print('cost: ',time_end-time_start,' s')
    for i in range(jm.shape[0]):
        writeline = [class_list[i]]
        for j in range(jm.shape[1]):
            writeline.append(jm[i,j])
        f2_csv.writerow(writeline)
    # print(jm)
    # print(np.isnan(jm))
    f2_csv.writerow(['final JM-Score: ',score])
    print(score)
    f2.close()