from BSNET_Conv import *
# import os
# import numpy as np
# import random
from utils.newDataset import *
from utils.os_helper import mkdir
from utils.parse_args import parse_args
import math
import sys
from materialNet import MaterialSubModel
from model.DBDA_Conv import DBDA_network_MISH_full_conv, DBDA_network_MISH_without_bn, DBDA_network_MISH
from torch.autograd import Variable
# from sklearn import preprocessing
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# CUDA:0

# mBatchSize = 32 #尽量能够被训练总数整除
# mEpochs = 300
select_bands_num = 18
# all_bands_nums = 128
all_bands_nums = 18
# class_nums = 4
# nora = False
#mLearningRate = 0.001
# mLearningRate = 0.0001
# mDevice=torch.device("cuda")
# model_path = './bs_model_sh/'
# device = torch.device('cuda')
args = parse_args()
# batchsize 也可以改 目前为32
mBatchSize = args.batchsize

mEpochs = args.epoch
model_select = args.model_select
mLearningRate = args.lr
# 梯度更新步长 0.0001
mDevice=torch.device("cuda")
nora = args.nora
print('mBatchSize',mBatchSize)
print('mEpochs',mEpochs)
print('mLearningRate',mLearningRate)
print('model_select',model_select)
print('nora',nora)
# model_path = ['./IntervalSampleWaterModel/','./newSampleWaterModel/','./IntervalSampleBigWaterModel/']
model_size = ["oriMaterial", "DBDA"]
featureTrans = args.featureTrans#False#
# model_save = model_path[model_select-1]

# model_save = './IntervalSampleAddFeatureWaterModel_shenzhen2/'
# dataTypelist = ['water', 'water','water']
# dataType = dataTypelist[model_select-1]
dataType = 'sea'
print('featureTrans',featureTrans)
# select_bands = [2,36,54,61,77,82,87,91,95,104,108]
# select_bands = [x + 5 for x in  select_bands]

train_bands = [x for x in range(all_bands_nums)]

# hashBand = hash(select_bands)
# 换成小模型 11 通道 重新验证一下？？
# select_bands = [x for x in range(128)]

class_nums = 3

if featureTrans:
    bands_num = 21
else:
    bands_num = len(train_bands)

if bands_num != 128:
    bands_str = [str(x) for x in train_bands]
    bands_str = "_".join(bands_str)
else:
    bands_str = "128bands"

intervalSelect = args.intervalSelect
print('intervalSelect :', intervalSelect)

model_save = "./modelSave/" + model_size[model_select - 1] + "_add_bj_uneven_" + str(mBatchSize) + "_" + str(mLearningRate) + "/"

mkdir(model_save)

per_label_nums_per_img = args.per_label_nums_per_img
print("per_label_nums_per_img: ", per_label_nums_per_img)

if __name__ == '__main__':
    seed = 2021
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    # 可以考虑改变benchmark为true
    torch.backends.cudnn.benchmark = False
    # 配合随机数种子，确保网络多次训练参数一致
    torch.backends.cudnn.deterministic = True
    # 使用非确定性算法
    torch.backends.cudnn.enabled = True

    # 直接 输入 128 通道原始数据 吧 因为 网络第一层就是BN层！！！
    # if os.path.exists('./trainData.npy'):
    #     trainData = np.load('./trainData.npy')
    #     trainLabel = np.load('./trainLabel.npy')
    # else:
    #     trainData, trainLabel = generateData('all', 100, 11, DATA_TYPE, nora=nora)
    #     try:
    #         np.save('trainData.npy',trainData)
    #         np.save('trainLabel.npy', trainLabel)
    #     except Exception as e:
    #         print("error:", e)
    #         # pass
    save_npy_path = './trainData/'
    save_trainData_npy_path = save_npy_path + bands_str + str(intervalSelect) + "_" + str(per_label_nums_per_img) + '_mulprocess.npy'
    mkdir(save_npy_path)
    # save_trainData_npy_path = './trainData/big_32_0.001_Falsemulprocess1.npy'
    print(save_trainData_npy_path)
    # multiProcessGenerateData(dataType, num, length, nora=True, class_nums=2, intervalSelect=True, featureTrans=True)
    # trainData, trainLabel = multiProcessGenerateData(dataType, 2500, 11, nora=nora, class_nums=class_nums,
    #                                      intervalSelect=True, featureTrans=featureTrans)
    # try:
    #     np.save(save_trainData_npy_path, trainData)
    #     np.save(save_trainData_npy_path[:-4] + '_label.npy', trainLabel)
    # except:
    #     print("error")
    #     sys.exit()
    # sys.exit()

    # if not os.path.exists(save_trainData_npy_path):
    #     # 数据的归一化 应该在分割完patch之后 避免以后需要不归一化的数据
    #     trainData, trainLabel = multiProcessGenerateData(dataType, per_label_nums_per_img, 11, train_bands, 'sig', nora= False, class_nums=class_nums,
    #                                                      intervalSelect=intervalSelect, featureTrans=False)
    #     # trainData, trainLabel = generateData("dataType", 2500, 11, DATA_TYPE, nora=nora, class_nums=class_nums, intervalSelect = True, featureTrans = featureTrans)
    # # trainData, trainLabel = generateData(dataType, 1000, 11, DATA_TYPE,nora=nora, class_nums = class_nums)
    #
    # # testData, testLabel = generateData(dataType, 600, 11, DATA_TYPE,nora=nora, class_nums=class_nums)
    # # trainData = np.load('./trainData/train.npy')
    # # trainLabel = np.load('./trainData/trainLabel.npy')
    # # trainData = np.load('./trainData/trainIntervalAddFeature_1.npy')
    # # trainLabel = np.load('./trainData/trainLabelIntervalAddFeature_1.npy')
    #     if not os.path.exists(save_trainData_npy_path):
    #         try:
    #             np.save(save_trainData_npy_path,trainData)
    #             np.save(save_trainData_npy_path[:-4] + '_label.npy', trainLabel)
    #             # np.save('./testData/testData.npy',testData)
    #             # np.save('./testData/testLabel.npy',testLabel)
    #         except BaseException as e:
    #             print("path not exist error!!!!!")
    #             print(e)
    #             sys.exit()
    # else:
    #     trainData = np.load(save_trainData_npy_path)
    #     trainLabel = np.load(save_trainData_npy_path[:-4] + '_label.npy')
    #     print("train data exist!!!")
    trainData = np.load('/home/cjl/ywj_code/code_hz18/trainData3_addbj_uneven_allband' + str(True) + '_' + str(3) + '.npy')
    trainLabel = np.load('/home/cjl/ywj_code/code_hz18/trainLabel3_addbj_uneven_allband' + str(True) + '_' + str(3) + '.npy')
    print("begin!!!")
    print("trainData shape : ", trainData.shape)
    print("trainLabel shape : ", trainLabel.shape)
    # #trainData B C H W 要在 HW 方向归一化数据 好像也没啥必要 通道 归一化 分类 或者 不进行预处理 直接 分类？
    # trainData = trainData.transpose(0, 2, 3, 1)  # BHW C
    # #
    # trainData_ = trainData.reshape(np.prod(trainData.shape[:3]), np.prod(trainData.shape[3:]))
    # # trainData = trainData.reshape(trainData.shape[0]*trainData.shape[1],trainData.shape[2]*trainData.shape[3])
    # scaler = preprocessing.MinMaxScaler()
    # trainData_ = scaler.fit_transform(trainData_)
    # trainData = trainData_.reshape(trainData.shape)
    #
    # trainData = trainData.transpose(0, 3, 1, 2)  # B C HW
    # #
    # print(scaler.data_min_, scaler.data_max_)

    n_sam = trainData.shape[0]
    print('number of training',n_sam)
    #
    # # sys.exit()
    # testData, testLabel = generateData('test', 300, 11, DATA_TYPE, nora=nora)
    # testData = testData.transpose(0, 2, 3, 1)  # BHW C
    # testData_ = testData.reshape(np.prod(testData.shape[:3]), np.prod(testData.shape[3:]))
    # testData_ = scaler.transform(testData_)
    # testData = testData_.reshape(testData.shape)

    # 重新生成训练数据 按照切面均值归一化来 直接照搬DBDA的那个数据预处理就好了 或者换成minmaxScaler 根本不用分训练集和测试集 直接所有数据
    trainDataset = MyDataset(trainData, trainLabel)
    # testDataset = MyDataset(testData, testLabel)
    trainLoader = DataLoader(dataset=trainDataset, batch_size=mBatchSize, shuffle=True)
    # testLoader = DataLoader(dataset=testDataset, batch_size=mBatchSize, shuffle=True)
    if model_select == 1:
        model = BSNET_Embedding(all_bands_nums, class_nums, MaterialSubModel, False).cuda()
    else:
        model = BSNET_Embedding(all_bands_nums, class_nums, DBDA_network_MISH, True).cuda()
    # model.load_state_dict(torch.load(r"/home/yuwenjun/lab/multi-category/model_sh_hz/9.pkl"))
    # criterion=nn.MSELoss()
    # criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=mLearningRate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=6,
                                                           verbose=True, threshold=0.005, threshold_mode='rel',
                                                           cooldown=0,
                                                           min_lr=0, eps=1e-08)
    channel_weight_list = []
    channel_weight_list_test = []
    loss_list = []
    test_list = []
    for epoch in range(mEpochs):
        # 训练
        model.train()
        trainTotal = 0
        trainCorrect = 0
        trainLossTotal = 0.0
        for i, data in enumerate(trainLoader, 0):
            img, label = data
            # img = Variable(img).float().cuda()
            img, label = Variable(img).float().to(mDevice), Variable(label).long().to(mDevice)
            img_caloss = img.clone()
            predict, weight = model(img)
            predict = predict.squeeze()
            predictIndex = torch.argmax(predict, dim=1)
            labelIndex = torch.argmax(label, dim=1)
            trainCorrect += (predictIndex == labelIndex).sum()
            trainTotal += label.size(0)
            # 产生loss
            loss = criterion(predict, labelIndex) + 0.01 * torch.sum(torch.abs(weight))
            trainLossTotal += loss.item()
            # print("loss = %.5f" % float(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss = criterion(predict, img) + 0.01*torch.sum(weight)/img.size()[0]# 对权重实施l1正则化，并除以批量数，原论文没有除批量数
            # weight 激活值为0-1之间，L1正则化不用加绝对值
            # loss = criterion(predict, img_caloss) + 0.01 * torch.sum(torch.abs(weight))
        tra_acc = trainCorrect.item() / trainTotal
        # print('train epoch loss:', epoch, ': ',trainLossTotal)
        print('train epoch:', epoch, ' loss : ', float(trainLossTotal), ' ; acc : ', tra_acc)

        scheduler.step(trainLossTotal)
        trainLossTotal = 0
        trainTotal = 0
        trainCorrect = 0
        #一个epoch训练完后
        model.eval()
        weight_batch = np.zeros((n_sam, all_bands_nums))
        batch_val = mBatchSize*256
        cnt = math.ceil(n_sam / batch_val)
        for i in range(cnt):
            imgData = trainData[i * batch_val:(i + 1) * batch_val if (i + 1) < cnt else n_sam]
            label = trainLabel[i * batch_val:(i + 1) * batch_val if (i + 1) < cnt else n_sam]  # b,4
            imgData = np.array(imgData)
            label = np.array(label)
            inputData = torch.tensor(imgData).float().cuda()
            label = torch.tensor(label).float().cuda()
            with torch.no_grad():
                img_caloss = inputData.clone()
                predict ,weight = model(inputData) #b,c,h,w--b,4,1,1
                predict = predict.squeeze()  # b,4
                predictIndex = torch.argmax(predict, dim=1)
                labelIndex = torch.argmax(label, dim=1)
                trainCorrect += (predictIndex == labelIndex).sum()
                trainTotal += label.size(0)
                # print(predict.shape)
                # print(labelIndex.shape)
                loss = criterion(predict, labelIndex) + 0.01 * torch.sum(torch.abs(weight))
                # loss = criterion(predict, img_caloss) + 0.01*torch.sum(torch.abs(weight))# 对权重实施l1正则化，并除以批量数
                weight = weight.squeeze() #B*C
                weight_ = weight.detach().cpu().numpy()
                weight_batch[i * batch_val:(i + 1) * batch_val if (i + 1) < cnt else n_sam] = weight_
                trainLossTotal += loss.item()
        tra_acc = trainCorrect.item() / trainTotal
        # print('train epoch loss:', epoch, ': ', trainLossTotal)
        # 为什么 第二个loss比第一个小 因为 batchsize 变成了第一个的256倍数 而 criterion 本身是取了均值的  reduction 默认 是 mean
        print('train epoch:', epoch, ' loss : ', float(trainLossTotal), ' ; acc : ', tra_acc)
        channel_weight_list.append(weight_batch)
        loss_list.append(trainLossTotal)
        print('\n')
        # 每个 像素块 patch 的 各个通道的权值 求平均 得到整体 每个通道的权重！！！
        mean_weight = np.mean(weight_batch, axis=0)
        band_indx = np.argsort(mean_weight)[::-1][:select_bands_num]
        print("the mean weight of every channel : ", np.sort(mean_weight)[::-1])
        print('train epoch:', epoch, ', select best bands:',band_indx)
        print('\n')
        # if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), model_save + str(epoch) + '.pkl')

        # model.eval()
        # weight_batch2 = np.zeros((n_sam2, bands))
        # batch_val = mBatchSize * 2
        # cnt = math.ceil(n_sam2 / batch_val)
        # testLossTotal =0
        # for i in range(cnt):
        #     file_tmp = test_list[i * batch_val:(i + 1) * batch_val if (i + 1) < cnt else n_sam2]
        #     imgData = []
        #     for filename in file_tmp:
        #         imgData_tmp = np.load(test_path + filename)
        #         imgData.append(imgData_tmp)
        #     imgData = np.array(imgData)
        #     inputData = torch.tensor(imgData).float().cuda()
        #     with torch.no_grad():
        #         predict, weight = model(inputData)  # b,c,h,w--b,4,1,1
        #         loss = criterion(predict, inputData) + 0.01 * torch.sum(weight) / img.size()[0]  # 对权重实施l1正则化，并除以批量数
        #         weight.squeeze()  # B*C
        #         weight_ = weight.detach().cpu().numpy()
        #         weight_batch2[i * batch_val:(i + 1) * batch_val if (i + 1) < cnt else n_sam] = weight_
        #         testLossTotal += loss.item()
        # print('test epoch:', epoch, ': ', testLossTotal)
        # test_list.append(testLossTotal)
        # channel_weight_list_test.append(weight_batch2)

        # print('\n')
        # mean_weight = np.mean(weight_batch2, axis=0)
        # band_indx = np.argsort(mean_weight)[::-1][:select_num*3]
        # print('test epoch:', epoch, ', select best bands:', band_indx)

    # shape : epoch, BatchSize, Channel
    # np.float64 内存要爆
    channel_weight_list = np.array(channel_weight_list, dtype=np.float32)
    np.save('channel_weight_list.npy',channel_weight_list)
    # channel_weight_list_test = np.array(channel_weight_list_test)
    # np.save('channel_weight_list_test.npy', channel_weight_list_test)