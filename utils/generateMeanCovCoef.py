# from bsUtils import SequentialFeatureSelectionFourClass,SBSFourClass, get_average_jm_score_four_class
# import csv
import time
import numpy as np
import random
from tqdm import tqdm

if __name__ == "__main__":
    select_num = 9
    # cloth = [0, 1, 2, 4, 5, 6]
    # other = [3]
    # other.extend([x for x in range(8, 29)])
    # skin = 7
    '''
    npypath = '../trainData/'
    # select_num = 9
    mean_v = np.load(npypath + 'mean.npy')
    print(mean_v.shape)
    cov_v = np.load(npypath + 'cov.npy')
    print(cov_v.shape)
    twoBandJMScore = np.zeros(cov_v.shape[1:],dtype=np.float64)
    for b1 in tqdm(range(127)):
        for b2 in range(b1+1,128):
            m = mean_v[:,[b1,b2]]
            v = cov_v[:, [b1, b2]]
            v = v[:,:,[b1, b2]]
            twoBandJMScore[b1][b2] = get_average_jm_score_four_class(m, v)
            twoBandJMScore[b2][b1] = twoBandJMScore[b1][b2]
    print(twoBandJMScore)
    np.save(npypath + 'twoBandJMScore.npy', twoBandJMScore)
    R = np.zeros_like(cov_v)
    # 协方差矩阵与相关系数矩阵变换原理 ： https://www.zhihu.com/question/469872088
    for cla in range(mean_v.shape[0]):
        D = np.diag(np.sqrt(np.diag(cov_v[cla])))  # 提取cov_X的对角线元素构成一个对角矩阵
        D_inv = np.linalg.inv(D)  # 求D的逆矩阵
        R[cla] = D_inv @ cov_v[cla] @ D_inv  # 系数矩阵 叉乘
    print(R)
    mean_R = np.nanmean(R, axis=0)
    print(1/mean_R)
    print(mean_R.shape)
    np.save(npypath + 'corrcoef.npy',R)
    '''

    trainDataNpy = '../trainData/128bandsFalse_60_31class.npy'
    trainLabelNpy = '../trainData/128bandsFalse_60_31class_label.npy'
    savepath = '../trainData/'
    trainData = np.load(trainDataNpy)
    trainLabel = np.load(trainLabelNpy)

    print('trainData type:', trainData.dtype)
    print('trainData shape:', trainData.shape)  # b c h w
    print('trainLabel shape:', trainLabel.shape)  # b class_num one-hot形式
    trainLabel = np.nanargmax(trainLabel, axis=1)
    print('trainLabel shape:', trainLabel.shape)
    trainData = trainData.transpose(0, 2, 3, 1)  # b h w c
    print('trainData shape:', trainData.shape)  # b h w c

    # 重新分割一下，类别均衡来选取，重新计算一下！！
    spec_num = 128
    # class_num = 4  # 0:other 1:skin 2:cloth 3:water
    class_num = 31  # 0:water 30:other
    pixall = [np.empty([0, spec_num], dtype=np.float32) for _ in range(class_num)]  # 类别要注意对应
    # 类别均衡选取
    class_nums = []
    for i in range(class_num):
        class_nums.append(np.sum(trainLabel == i))

    selected_nums = np.min(np.array(class_nums))  # 每个类别选取这么多个样本

    for i in range(class_num):
        i_pos_index = np.where(trainLabel == i)  # trainLabel是一维，所以返回一个一维array
        i_pos_index = list(i_pos_index[0])
        # print(i_pos_index)
        # print(len(i_pos_index))
        # # pos_list = list(range(len(i_pos_index)))
        # # select_pos = random.sample(pos_list, selected_nums)
        # # select_index = i_pos_index[select_pos]
        # select_index = random.sample(i_pos_index, selected_nums)
        # print(select_index)
        pixall[i] = np.concatenate((pixall[i], trainData[i_pos_index, 5, 5, :]), axis=0)
        # print(pixall[i])


    # for order, label in enumerate(trainLabel):  # 0 1 2 3
    #     pixall[label] = np.append(pixall[label], trainData[order, 5, 5:6, :], axis=0)

    mean_v = [np.nanmean(specdata, axis=0) for specdata in pixall]  # n,x,y n为类别数 b为样本数量 y为通道数量 是基于像素点选取的！
    mean_v = np.array(mean_v)  # shape: n, spec_num
    print(mean_v.shape)
    print(mean_v)
    # np.cov 计算时，spec_num应该是第一维度！！
    cov_v = [np.cov(specdata.transpose(1, 0), ddof=1) for specdata in pixall]  # shape: n,spec_num,spec_num
    cov_v = np.array(cov_v)
    print(cov_v.shape)
    print(cov_v)
    np.save(savepath + 'mean31class.npy', mean_v)
    np.save(savepath + 'cov31class.npy', cov_v)
