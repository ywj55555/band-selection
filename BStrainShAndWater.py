from torch.autograd import Variable

from BSNET_Conv import *
import os
import numpy as np
import random
from utils.newDataset import *
from utils.os_helper import mkdir
import math
from sklearn import preprocessing
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# CUDA:0

mBatchSize = 32 #尽量能够被训练总数整除
mEpochs = 300
select_num = 27
bands = 128
nora = False
#mLearningRate = 0.001
mLearningRate = 0.0001
mDevice=torch.device("cuda")
model_path = './modelSave/bsNets_shAndWater_' + str(mBatchSize) + "_" + str(mLearningRate) + "/"

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
    mkdir(model_path)
    traindatanpy = '/home/cjl/ywj_code/graduationCode/BS-NETs/trainData/128bandsFalse_60_mulprocess.npy'
    trainlabelnpy = '/home/cjl/ywj_code/graduationCode/BS-NETs/trainData/128bandsFalse_60_mulprocess_label.npy'
    if os.path.exists(traindatanpy):
        trainData = np.load(traindatanpy)
        trainLabel = np.load(trainlabelnpy)
    else:
        trainData, trainLabel = generateData('all', 100, 11, DATA_TYPE, nora=nora)
        try:
            np.save('trainData.npy',trainData)
            np.save('trainLabel.npy', trainLabel)
        except Exception as e:
            print("error:", e)
            # pass
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
    trainData = trainData / np.max(trainData, axis=1, keepdims=True)  # 通道归一化
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

    model = BSNET_Conv(bands).cuda()
    # model.load_state_dict(torch.load(r"/home/yuwenjun/lab/multi-category/model_sh_hz/9.pkl"))
    # criterion=nn.MSELoss()
    criterion = nn.MSELoss()
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
        trainLossTotal = 0.0
        for i, data in enumerate(trainLoader, 0):
            img, _ = data
            img = Variable(img).float().cuda()
            img_caloss = img.clone()
            predict ,weight = model(img)
            # loss = criterion(predict, img) + 0.01*torch.sum(weight)/img.size()[0]# 对权重实施l1正则化，并除以批量数，原论文没有除批量数
            # weight 激活值为0-1之间，L1正则化不用加绝对值
            loss = criterion(predict, img_caloss) + 0.01 * torch.sum(torch.abs(weight))
            trainLossTotal += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('train epoch:', epoch, ': ',trainLossTotal)
        scheduler.step(trainLossTotal)
        trainLossTotal = 0
        #一个epoch训练完后
        model.eval()
        weight_batch = np.zeros((n_sam, bands))
        batch_val = mBatchSize*256
        cnt = math.ceil(n_sam / batch_val)
        for i in range(cnt):
            imgData = trainData[i * batch_val:(i + 1) * batch_val if (i + 1) < cnt else n_sam]
            # imgData = []
            # for filename in file_tmp:
            #     imgData_tmp = np.load(train_path + filename)
            #     imgData.append(imgData_tmp)
            imgData = np.array(imgData)
            inputData = torch.tensor(imgData).float().cuda()
            with torch.no_grad():
                img_caloss = inputData.clone()
                predict ,weight = model(inputData) #b,c,h,w--b,4,1,1
                loss = criterion(predict, img_caloss) + 0.01*torch.sum(torch.abs(weight))# 对权重实施l1正则化，并除以批量数
                weight = weight.squeeze() #B*C
                weight_ = weight.detach().cpu().numpy()
                weight_batch[i * batch_val:(i + 1) * batch_val if (i + 1) < cnt else n_sam] = weight_
                trainLossTotal += loss.item()
        print('train epoch:', epoch, ': ', trainLossTotal)
        channel_weight_list.append(weight_batch)
        loss_list.append(trainLossTotal)
        print('\n')
        mean_weight = np.mean(weight_batch, axis=0)
        band_indx = np.argsort(mean_weight)[::-1][:select_num]
        print('train epoch:', epoch, ', select best bands:',band_indx)
        print('\n')
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), model_path + str(epoch) + '.pkl')

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

    channel_weight_list = np.array(channel_weight_list)
    np.save('BS_channel_weight_list.npy',channel_weight_list)
    # channel_weight_list_test = np.array(channel_weight_list_test)
    # np.save('channel_weight_list_test.npy', channel_weight_list_test)