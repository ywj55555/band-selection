import gc
import random
from skimage import io
import math
from multiprocessing import Process,Pool
import multiprocessing as mp
# import numpy as np
from data.dictNew import *
from utils.load_spectral import *
import os
import copy
# from sklearn.preprocessing import LabelEncoder
# from torch.autograd import Variable
# import torch

# code2label = [0,2,2,2,0,2,2,2,1,0,0,0,0,0,0,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0] #1:skin 2:cloth 3:plant 0:other
code2labelSh = [0,2,2,2,0,2,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #1:skin 2:cloth 3:plant 0:other
# label2class = [255,1,2,0]
code2labelWater1 = [30, 0]  #
code2labelWater2 = [255, 0, 30]  # #1:skin 2:cloth 0:water 30:other
label2class = [255,1,0]
label2target = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
class_nums_jm = 31

SKIN_TYPE = 0
CLOTH_TYPE = 1
PLANT_TYPE = 2

# DATA_TYPE = SKIN_TYPE
# DATA_TYPE = CLOTH_TYPE
DATA_TYPE = PLANT_TYPE

SELECT_ALL = 0  # select all pixels for training and testing
SELECT_RANDOM = 1  # select pixels randomly for training and testing
#
# env_data_dir3 = '/home/cjl/dataset/envi/'
# label_data_dir = '/home/cjl/dataset/label/'
waterImgPath = '/home/cjl/waterDataset/TO_WenJun/'
waterLabelPath = '/home/cjl/waterDataset/TO_WenJun/label/'
shImgPath = '/home/cjl/dataset/envi/'
shLabelPath = '/home/cjl/dataset/label/'

video3_img_path = 'D:/ZF2121133HHX/20220407/vedio3/'
video6_img_path = 'D:/ZF2121133HHX/20220407/vedio6/'

video3_label_path = 'D:/ZF2121133HHX/20220407/video3_labels/'
video6_label_path = 'D:/ZF2121133HHX/20220407/video6_labels/'

# waterLabelPath = '/home/cjl/ssd/dataset/shenzhen/label/Label_rename/'

waterImgRootPath = '/home/cjl/ssd/dataset/shenzhen/img/train/'
# waterImgRootList = os.listdir(waterImgRootPath)
# waterImgRootList = [x for x in waterImgRootList if x[-4:] == '.img']
waterImgRootPathList = ['vedio1','vedio2','vedio3','vedio4','vedio5','vedio6','vedio7']
# label_data_dir_6se = '/home/cjl/data/sensor6/label/'
# tif_dir = '/home/cjl/data/sensor6/tif_data/'
#env_data_dir = 'E:/BUAA_Spetral_Data/hangzhou/envi/'
#label_data_dir = 'E:/BUAA_Spetral_Data/hangzhou/label/'

# 每一类 取样最小间隔
min_interval = [4,10]
# 记录每张图 每一个类 取样了多少patch
class_nums_record = {1:0,2:0}

output_log = open('./log/output_generate.log','w')

# def generateData(dataType, num, length, selectMode=SELECT_RANDOM,nora = True,allband=True,feature=False,class_num=10,cut_num=10,per_class_num=None):
#     Data = []
#     Label = []
#     # if dataType == 'train':
#     #     dataFile = trainFile
#     # elif dataType == 'test':
#     #     dataFile = testFile
#     if dataType == 'train':
#         dataFile = trainFile
#     elif dataType == 'test':
#         dataFile = testFile
#     else:
#         dataFile = testfile
#     for file in dataFile:
#
#         # t1 = time.time()
#         imgLabel = io.imread(label_data_dir + file + '.png')
#         imgLabel = imgLabel[cut_num:-cut_num,cut_num:-cut_num]
#         # t2 = time.time()
#         if not os.path.exists(env_data_dir+file+'.raw'):
#             imgData = raw_loader(env_testdata_dir, file, nora=nora, allband=allband, feature=feature,cut_num=cut_num)
#         else:
#             imgData = raw_loader(env_data_dir,file,nora=nora,allband=allband,feature=feature,cut_num=cut_num)
#         # imgData = transform2(imgData)
#         if file in add_other_nums:
#             tmp_per_class_num = [500,500,7500]
#             print(file, tmp_per_class_num)
#             pix, label = generateAroundDeltaPatchAllLen(imgData, imgLabel, num, length, selectMode,class_num=class_num,per_class_num=tmp_per_class_num)
#         else:
#             print(file, per_class_num)
#             pix, label = generateAroundDeltaPatchAllLen(imgData, imgLabel, num, length, selectMode, class_num=class_num,
#                                                         per_class_num=per_class_num)
#         Data.extend(pix)
#         Label.extend(label)
#
#     return Data, Label
def generateData(dataType, num, length, typeCode, selectMode=SELECT_RANDOM,nora = True, class_nums = 2, intervalSelect = True, featureTrans = True):
    Data = []
    Label = []
    if dataType == 'sea':
        dataFile = SeaFile
        labelpath = waterLabelPath
        imgpath = waterImgRootPath
    # if dataType == 'train':
    #     dataFile = trainFile
    # elif dataType == 'test':
    #     dataFile = testFile
    # if dataType == 'water':
    #     dataFile = waterFile
    #     labelpath = waterLabelPath
    #     imgpath = waterImgRootPath
    # elif dataType == 'water':
    #     dataFile = waterFile
    #     imgpath = waterImgRootPath
    #     labelpath = waterLabelPath
    else:
        dataFile = waterFileTest
        labelpath = waterLabelPath
        imgpath = waterImgRootPath
    for file in dataFile:
        print(file)
        # 以rgb开头
        # t1 = time.time()
        # 读取标签文件
        # 1 skin 2 cloth 3 other
        imgLabel = io.imread(labelpath + file + '.png')
        # t2 = time.time()
        # imgData = envi_loader(os.path.join(env_data_dir,file[:8])+'/', file,nora)
        # 读取img文件
        imgData = None
        if os.path.exists(imgpath + file[3:] + '.img'):
            imgData = envi_loader(imgpath, file[3:], False)
        else:
            for tmpImgPath in waterImgRootPathList:
                if os.path.exists(imgpath + tmpImgPath + '/' + file[3:] + '.img') :
                    imgData = envi_loader(imgpath + tmpImgPath + '/', file[3:], False)
                    break
        # t3 = time.time()
        # 没必要 特征变换 增加之前设计的斜率特征
        if imgData is None:
            print("Not Found ",file)
            continue
        # H W 22
        if featureTrans:
            print("kindsOfFeatureTransformation......")
            # 11 -》 21
            imgData = kindsOfFeatureTransformation(imgData, nora)
        else:
            if nora:
                print("normalizing......")
                imgData = envi_normalize(imgData)

        # t4 = time.time()
        # print(imgData.shape)
        # if imgData.shape != (1415, 1859, 22):
        #     continue
        # print(typeCode)
        # 11*11像素块 类别预测 作为中心像素类别 可以利用上下文信息 提升准确率
        # 分割出像素块 返回像素块和对应的类别
        if intervalSelect:
            pix, label = intervalGenerateAroundDeltaPatchAllLen(imgData, imgLabel, num, length, file, 50 ,
                                                        class_nums)
        else:
            pix, label = generateAroundDeltaPatchAllLen(imgData, imgLabel, typeCode, num, length, selectMode,class_nums)
        #print(pix.shape)
        #print(label.shape)
        # t5 = time.time()
        # print('read label:', t2 - t1, ',read envi:', t3 - t2, ',transform:', t4 - t3, ',genrate:', t5 - t4)
        # read label: 3.676800489425659 ,read envi: 9.025498867034912 ,transform: 0.22420549392700195 ,genrate: 2.00569748878479
        Data.extend(pix)
        Label.extend(label)
    return Data, Label

def singleProcessGenerateData(dataType, dataFile, num, length, bands, activate, nora = False, class_nums = 2, intervalSelect = True, featureTrans = False):
    id = os.getpid()
    print("当前进程id: ", id, "begin!!!!!!")
    Data = []
    Label = []
    if dataType == 'sea':
        labelpath = waterLabelPath
        imgpath = waterImgRootPath
    elif dataType == 'shAndWater':
        # dataFile = shDataAndWater
        labelpath = None
        imgpath = None
    else:
        labelpath = waterLabelPath
        imgpath = waterImgRootPath
    dataFile_LEN = len(dataFile)
    for file_order,file in enumerate(dataFile):
        print(file)
        print(file_order, "/" ,dataFile_LEN)
        output_log.writelines(str(file_order)+ "/" + str(dataFile_LEN))
        # 1 skin 2 cloth 3 other
        # 注意类别转换！！水体转为3类别 水体标签转为0和3
        if dataType == 'shAndWater':
            if os.path.exists(shLabelPath + file + '.png'):
                imgLabel = io.imread(shLabelPath + file + '.png')
            elif os.path.exists(waterLabelPath + 'rgb' + file + '.png'):
                imgLabel = io.imread(waterLabelPath + 'rgb' + file + '.png')
            else:
                print(waterLabelPath + 'rgb' + file + '.png')
                print(file, 'label not exist!!!')
        else:
            imgLabel = io.imread(labelpath + file + '.png')
        # t2 = time.time()
        # imgData = envi_loader(os.path.join(env_data_dir,file[:8])+'/', file,nora)
        # 读取img文件
        imgData = None
        if dataType == 'shAndWater':
            if os.path.exists(shLabelPath + file + '.png'):
                imgData = envi_loader(shImgPath + file[:8] + '/', file, bands, False)
            elif os.path.exists(waterLabelPath + 'rgb' + file + '.png'):
                imgData = envi_loader(waterImgPath, file, bands, False)
            else:
                # print()
                print(file, 'img not exist!!!')
        else:
            imgData = envi_loader(imgpath, file, bands, False)
        # if os.path.exists(imgpath + file[3:] + '.img'):
        #     imgData = envi_loader(imgpath, file[3:], bands, False)
        # else:
        #     for tmpImgPath in waterImgRootPathList:
        #         if os.path.exists(imgpath + tmpImgPath + '/' + file[3:] + '.img') :
        #             imgData = envi_loader(imgpath + tmpImgPath + '/', file[3:], bands,False)
        #             break
        # t3 = time.time()
        # 没必要 特征变换 增加之前设计的斜率特征
        if imgData is None:
            print("img Not Found ",file)
            continue
        # H W 22
        if featureTrans:
            print("kindsOfFeatureTransformation......")
            output_log.writelines("kindsOfFeatureTransformation......")
            # 11 -》 21
            imgData = kindsOfFeatureTransformation_slope(imgData,activate, nora)
        else:
            if nora:
                print("normalizing......")
                output_log.writelines("normalizing......")
                imgData = envi_normalize(imgData)
        # 11*11像素块 类别预测 作为中心像素类别 可以利用上下文信息 提升准确率
        # 分割出像素块 返回像素块和对应的类别
        if intervalSelect:
            print("intervalGenerateAroundDeltaPatchAllLen...")
            pix, label = intervalGenerateAroundDeltaPatchAllLen(imgData, imgLabel, num, length, file, 50, class_nums)
        else:
            print("random sampling...")
            if os.path.exists(shLabelPath + file + '.png'):
                # num : 60
                pix, label = generateAroundDeltaPatchAllLen(imgData, imgLabel, num, length, 30)
            else:
                # 选取水体的 # num : 60 * 3 = 180
                pix, label = generateAroundDeltaPatchAllLen(imgData, imgLabel, num * 3, length, 2)
            # pix, label = generateAroundDeltaPatchAllLen(imgData, imgLabel, num, length, class_nums)
        Data.extend(pix)
        Label.extend(label)
        del imgData
        gc.collect()
    # id = os.getpid()
    print("当前进程id: ", id , "done!!!!!!")
    return Data, Label

def multiProcessGenerateData(dataType, num, length, bands, activate, nora = True, class_nums = 2, intervalSelect = True, featureTrans = True):
    all_process_data = []
    all_process_label = []
    if dataType == 'sea':
        dataFile = SeaFile
        labelpath = waterLabelPath
        imgpath = waterImgRootPath
    elif dataType == 'shAndWater':
        dataFile = shDataAndWater
    else:
        dataFile = waterFileTest
        labelpath = waterLabelPath
        imgpath = waterImgRootPath
    # 启动多个进程 分发文件列表
    print('cpu nums : ', mp.cpu_count())
    process_nums = os.cpu_count()
    # print(mp.cpu_count())
    # process_nums = os.cpu_count()
    if process_nums > len(dataFile):
        process_nums = len(dataFile)
    id = os.getpid()
    # 获取当前进程id
    print("当前主进程id: ", id)
    # 进程池中从无到有创建process_nums个进程,以后一直是这process_nums个进程在执行任务 processes 参数默认值通过 os.cpu_count() 获取。
    process_poll = Pool(processes=process_nums)
    seg_data_all_process = []
    start = time.time()
    all_file_nums = len(dataFile)
    print("the len of dataFile is : ", all_file_nums)
    output_log.writelines("the len of dataFile is : " + str(all_file_nums))
    perProcess = math.ceil(all_file_nums / process_nums)
    for i in range(process_nums):
        print(i)  # for循环会提前运行完毕，进程池内的任务还未执行。
        # 划分问价列表
        left = i * perProcess
        right = (i + 1) * perProcess if (i + 1) * perProcess < all_file_nums else all_file_nums
        # args : (dataType, dataFile, num, length, nora = True, class_nums = 2, intervalSelect = True, featureTrans = True)
        # 同步运行,阻塞、直到本次任务执行完毕拿到 single_res

        single_res = process_poll.apply_async(singleProcessGenerateData, args=(dataType, dataFile[left:right], num, length, bands, activate, nora, class_nums, intervalSelect, featureTrans))
        seg_data_all_process.append(single_res)  # 将调用apply_async方法，得到返回进程内存地址结果
    process_poll.close()
    process_poll.join()
    # gc.collect()
    # gc.collect()
    # gc.collect()
    for single_res in seg_data_all_process:
        # print('Single_process!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        tmpres = single_res.get()
        # print(len(tmpres[0]))
        # print(tmpres[0][0].shape)
        # print('Single_process_end!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # tmpres 的 shape 可能不一致 所以
        all_process_data.extend(tmpres[0])
        all_process_label.extend(tmpres[1])
        del tmpres
        gc.collect()
    all_process_data = np.array(all_process_data, dtype=np.float32)
    all_process_label = np.array(all_process_label, dtype=np.int8)
    stop = time.time()
    print('seg data cost time is %s' % (stop - start))
    output_log.writelines('seg data cost time is %s' % (stop - start))
    output_log.close()
    return all_process_data, all_process_label

def convert_to_one_hot(y, C):
    return np.eye(C,dtype=np.int8)[y.reshape(-1)]

# 随机选取 存在的问题 选取可能不均匀 可能边缘，或者一些 比较分歧、重要的地方不能保证选取到，数据多样性就不能保证
def generateAroundDeltaPatchAllLen(imgData, imgLabel, labelCount=2000, length=11,class_nums = 2, overexposure_thre = 45000):
    row, col, d = imgData.shape
    imgData = np.array(imgData, np.float32)
    # H W C -> C H W
    print("move overexposure area....")
    max_mask = np.max(imgData, axis=2)
    imgLabel[max_mask > overexposure_thre] = 255  # 不能是0！！
    img = imgData.transpose(2, 0, 1) # c h w
    img = np.expand_dims(img, 0) # b c h w
    del max_mask
    gc.collect()
    # img.shape -> (1, channels, row, col)
    # 像素块变量 保存提取的像素块
    pix = np.empty([0, d, length, length], dtype=np.float32)
    labels = np.empty([0, class_nums_jm])  # 30 最后一个类别表示 other, 0表示水体
    length1 = int(length / 2)
    # 在这里进行分类！！
    # 30类 上海数据 2分类水体的 还有 1 分类的水体 根据最大值来进行分类！！
    if class_nums == 30:
        for label in range(1, class_nums + 1):  # 共有3类，取值为1、2、3、...、30，0为未标注
            # label_index = np.where(imgLabel == label)  # 找到标签值等于label的下标点 size：2*个数
            label_index = np.where(imgLabel[length1:-length1, length1:-length1] == label)
            pixels_nums = len(label_index[0])
            if pixels_nums == 0:
                continue
            # tmp = np.array(label2target[code2label[label]])
            # 获取对应类别 one-hot 标签
            # tmp = np.array(convert_to_one_hot(np.array([class_order for class_order in range(4)]), 4)[
            #                code2labelSh[label]])
            tmp = np.array(convert_to_one_hot(np.array([class_order for class_order in range(class_nums_jm)]), class_nums_jm)[
                               label])  # 1 - 30
            # 以下 用于 不均衡采样 类别样本
            per_class_num = [labelCount]
            per_class_num_care = [labelCount * 7, labelCount * 3] # 0 1 2 类别分别选取数量
            per_class_num.extend(per_class_num_care)
            for num in range(per_class_num[code2labelSh[label]]):
                index = random.randint(0, pixels_nums - 1)  # 前闭后闭区间 随机选取
                # labels = np.append(labels, label)
                #print(np.reshape(tmp, [-1, 4]), np.reshape(tmp, [-1, 4]).shape)
                labels = np.append(labels, np.reshape(tmp, [-1, class_nums_jm]), axis=0)
                x = label_index[0][index]
                y = label_index[1][index]
                # B C H W
                if length % 2 == 0:
                    pixAround = img[:, :, x : x + 2*length1, y : y + 2*length1]  # 3*3*4
                else:
                    pixAround = img[:, :, x: x + 2 * length1 + 1, y: y + 2 * length1 + 1]  # 3*3*4
                # 堆叠像素块
                pix = np.append(pix, pixAround, axis=0)
    elif np.max(imgLabel) == 2:
        for label in range(1, 3):  # 共有3类，取值为1、2,2为other,0为边界区域
            # label_index = np.where(imgLabel == label)  # 找到标签值等于label的下标点 size：2*个数
            label_index = np.where(imgLabel[length1:-length1, length1:-length1] == label)
            pixels_nums = len(label_index[0])
            if pixels_nums == 0:
                continue
            # tmp = np.array(label2target[code2label[label]])
            # 获取对应类别 one-hot 标签
            tmp = np.array(convert_to_one_hot(np.array([class_order for class_order in range(class_nums_jm)]), class_nums_jm)[
                               code2labelWater2[label]])
            # 以下 用于 不均衡采样 类别样本
            # per_class_num = [labelCount]
            # per_class_num_care = [labelCount * 3, labelCount * 2]  # 0 1 2 类别分别选取数量
            # per_class_num.extend(per_class_num_care)
            for num in range(labelCount):
                index = random.randint(0, pixels_nums - 1)  # 前闭后闭区间 随机选取
                # labels = np.append(labels, label)
                # print(np.reshape(tmp, [-1, 4]), np.reshape(tmp, [-1, 4]).shape)
                labels = np.append(labels, np.reshape(tmp, [-1, class_nums_jm]), axis=0)
                x = label_index[0][index]
                y = label_index[1][index]
                # B C H W
                if length % 2 == 0:
                    pixAround = img[:, :, x: x + 2 * length1, y: y + 2 * length1]  # 3*3*4
                else:
                    pixAround = img[:, :, x: x + 2 * length1 + 1, y: y + 2 * length1 + 1]  # 3*3*4
                # 堆叠像素块
                pix = np.append(pix, pixAround, axis=0)
    else:
        for label in range(0, 2):  # 共有2类，取值为0,1,0为未标注,1为水体
            # label_index = np.where(imgLabel == label)  # 找到标签值等于label的下标点 size：2*个数
            label_index = np.where(imgLabel[length1:-length1, length1:-length1] == label)
            pixels_nums = len(label_index[0])
            if pixels_nums == 0:
                continue
            # tmp = np.array(label2target[code2label[label]])
            # 获取对应类别 one-hot 标签
            tmp = np.array(convert_to_one_hot(np.array([class_order for class_order in range(class_nums_jm)]), class_nums_jm)[
                               code2labelWater1[label]])
            # 以下 用于 不均衡采样 类别样本
            # per_class_num = [labelCount]
            # per_class_num_care = [labelCount * 3, labelCount * 2]  # 0 1 2 类别分别选取数量
            # per_class_num.extend(per_class_num_care)
            for num in range(labelCount):
                index = random.randint(0, pixels_nums - 1)  # 前闭后闭区间 随机选取
                # labels = np.append(labels, label)
                # print(np.reshape(tmp, [-1, 4]), np.reshape(tmp, [-1, 4]).shape)
                labels = np.append(labels, np.reshape(tmp, [-1, class_nums_jm]), axis=0)
                x = label_index[0][index]
                y = label_index[1][index]
                # B C H W
                if length % 2 == 0:
                    pixAround = img[:, :, x: x + 2 * length1, y: y + 2 * length1]  # 3*3*4
                else:
                    pixAround = img[:, :, x: x + 2 * length1 + 1, y: y + 2 * length1 + 1]  # 3*3*4
                # 堆叠像素块
                pix = np.append(pix, pixAround, axis=0)
    labels = np.array(labels).astype('int64')
    pix = np.array(pix, dtype=np.float32)
    return pix, labels

def intervalGenerateAroundDeltaPatchAllLen(imgData, imgLabel, labelCount=2000, length=11, filename = 'null',thre=500, class_nums = 2, overexposure_thre = 45000):
    row, col,d = imgData.shape
    # img = np.expand_dims(imgData, 0)
    imgData = np.array(imgData)
    # H W C -> C H W
    # 去掉过曝区域
    print("move overexposure area....")
    max_mask = np.max(imgData, axis=2)
    imgLabel[max_mask > overexposure_thre] = 0
    img = imgData.transpose(2, 0, 1)
    del max_mask
    gc.collect()
    # B C H W
    img = np.expand_dims(img, 0)
    length1 = int(length / 2)
    pix = np.empty([0, d, length, length], dtype=float)
    labels = np.empty([0, class_nums])
    # 去掉过曝区域

    for label in range(1,class_nums + 1):  #label 已经转为0，1，2，3，4了 ,3植物 2衣物 1皮肤 4其他，0无标签
        # cnt = 1
        label_index = np.where(imgLabel[length1:-length1,length1:-length1] == label)  # 找到标签值等于label的下标点 size：2*个数
        pixels_nums = len(label_index[0])
        if pixels_nums <= thre:
            continue
        '''方案1
            1、如果 pixels_nums 小于 thre，说明目标过小，为无效数据，不采样；
            2、如果 pixels_nums < 4*labelCount ,则间隔行列采样 pixels_nums 不管数量；保证样本的稀疏性
            3、否则：
            计算pixels_nums是labelCount的多少倍，然后开立方根，向下取整，得到m，
            对label_index 进行按照行序号排序，间隔m-1采样 （从序号 m-1开始采样），开始序号为 (新的pixels_nums长度取模m +m-1)//2 (肯定小于m)
            如果满足，则结束，否则：
            对label_index 进行按照列序号排序，间隔m-1采样
            然后余下的为采样索引
            方案2-保证样本多样性(间隔取样)，根据该类数目选择间隔，设置一个最低和最高间隔
                在以上基础上尽可能保证数量均衡
                间隔取样：边界部分取样点可能相邻
                边界部分本来就难识别，多取样也并非坏事
        '''
        #最小采样间隔
        interval =  min_interval[label-1]
        #间隔1采样，数量除以4
        if pixels_nums > interval*interval*labelCount:
            mul = math.sqrt(pixels_nums/labelCount)
            interval = round(mul)
        print('before :',filename, 'label:',label, 'interval ',interval,'before :',pixels_nums)
        # print(label)
        # 也可以直接在原数组上操作，就不用排序索引了 10520没有进行采样
        # 一定要用深拷贝 可以了解一下 深拷贝 和 浅拷贝
        imgLabel_tmp = copy.deepcopy(imgLabel)
        # imgLabel_tmp = imgLabel
        # 隔行取样
        # cnt = imgLabel.shape[0]//interval
        cnt = math.ceil(imgLabel.shape[0] / interval)
        # print('cnt: ',cnt)
        mask = []
        for i in range(cnt):
            mask += [mk for mk in range(i*interval,interval*(i+1)-1 if interval*(i+1)-1<imgLabel.shape[0] else imgLabel.shape[0])]
            # imgLabel_tmp[i*cnt:cnt*(i+1)-1,:]=0
        # print('mask',mask)
        imgLabel_tmp[mask,:] = 0
        new_label_index = np.where(imgLabel_tmp == label)
        print('after row sampling: ',len(new_label_index[0]))
        # print(imgLabel_tmp)
        # 隔列取样
        mask = []
        # print(filename, label, interval)
        # cnt = imgLabel.shape[1] // interval
        cnt = math.ceil(imgLabel.shape[1] / interval)
        # print('cnt: ', cnt)
        for i in range(cnt):
            mask += [mk for mk in range(i*interval,interval*(i+1)-1 if interval*(i+1)-1<imgLabel.shape[1] else imgLabel.shape[1])]
        imgLabel_tmp[:,mask] = 0
        # print('mask', mask)
        #取样后的索引
        new_label_index = np.where(imgLabel_tmp[length1:-length1,length1:-length1] == label)
        #删掉边界像素！！
        # new_pixels_nums = len(new_label_index[0])
        # # 还要删掉边界位置
        # new_delete_index = []
        # for i in range(new_pixels_nums):
        #     if (new_label_index[0][i] <= length1 or new_label_index[0][i] >= row - length1 or new_label_index[1][
        #         i] <= length1 or
        #             new_label_index[1][i] >= col - length1):
        #         new_delete_index.append(i)
        # # 删除多列
        # new_label_index = np.delete(new_label_index, new_delete_index, axis=1)
        #取样所有label_index

        new_count = len(new_label_index[0])
        print('after sampling colounm',filename,label,new_count)
        # print(imgLabel_tmp)
        # continue
        # f.write(filename+' sampling '+str(label)+ ' class: '+str(new_count)+'\n')
        tmp = np.array(convert_to_one_hot(np.array([class_order for class_order in range(class_nums)]), class_nums)[
                           label2class[label]])

        global class_nums_record
        class_nums_record[label] += new_count
        for index in range(new_count):
            labels = np.append(labels, np.reshape(tmp, [-1, class_nums]), axis=0)
            x = new_label_index[0][index]
            y = new_label_index[1][index]
            pixAround = img[:,:,x : x + 2*length1 + 1, y : y + 2*length1 + 1]  # 21,21,bands
            pix = np.append(pix, pixAround, axis=0)
            # np.save(save_path + filename + name_list[label - 1] + str(index).rjust(4, '0') + '.npy', pixAround)
    labels = np.array(labels).astype('int64')
    pix = np.array(pix, dtype=float)
    # print(pix.shape)
    # print('label shape:',labels.shape)
    # 这个不是 单张图 的抽样结果吧！！！
    print(filename, "sampling :",class_nums_record)
    # B C H W
    return pix, labels

def generateAroundDeltaPatchAllLen_6se(imgData, imgLabel, typeCode, labelCount=2000, length=11, selectMode=SELECT_RANDOM):
    row, col, d = imgData.shape
    imgData = np.array(imgData)
    img = imgData.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    # img.shape -> (1, channels, row, col)
    pix = np.empty([0, d, length, length], dtype=float)
    labels = np.empty([0, 4])
    length1 = int(length / 2)
    for label in range(1, 5):  # 共有14类，取值为1、2、3、...、14
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

        tmp = np.array(label2target[label%4])

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

if __name__ == '__main__':
    for label in range(4):
        tmp = np.array(convert_to_one_hot(np.array([class_order for class_order in range(4)]), 4)[
                       label])
        print(tmp)
