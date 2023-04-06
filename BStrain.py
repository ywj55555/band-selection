from BSNET_Conv import *
import os
import numpy as np
import random
from utils.newDataset import *
from utils.os_helper import mkdir
import math
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# CUDA:0

mBatchSize = 32 #尽量能够被训练总数整除
mEpochs = 300
select_num = 9
bands = 128
#mLearningRate = 0.001
mLearningRate = 0.0001
mDevice=torch.device("cuda")
model_path = './bs_model_sh/'
train_path = "/data3/ywj/train_sh_hz_Patch_data/train/"
test_path = "/data3/ywj/train_sh_hz_Patch_data/test/"

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
    train_list = os.listdir(train_path)
    test_list = os.listdir(test_path)
    n_sam = len(train_list)
    n_sam2 =len(test_list)

    trainDataset = MyDataset(train_list, train_path)
    testDataset = MyDataset(test_list, test_path)
    # 重新生成训练数据 按照切面均值归一化来 直接照搬DBDA的那个数据预处理就好了 或者换成minmaxScaler 根本不用分训练集和测试集 直接所有数据


    trainLoader = DataLoader(dataset=trainDataset, batch_size=mBatchSize, shuffle=True)
    testLoader = DataLoader(dataset=testDataset, batch_size=mBatchSize, shuffle=True)

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
            predict ,weight = model(img)
            # loss = criterion(predict, img) + 0.01*torch.sum(weight)/img.size()[0]# 对权重实施l1正则化，并除以批量数，原论文没有除批量数
            # weight.squeeze()
            loss = criterion(predict, img) + 0.01 * torch.sum(weight)
            trainLossTotal += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('train epoch:', epoch, ': ',trainLossTotal)
        trainLossTotal = 0
        #一个epoch训练完后
        model.eval()
        weight_batch = np.zeros((n_sam, bands))
        batch_val = mBatchSize*2
        cnt = math.ceil(n_sam / batch_val)
        for i in range(cnt):
            file_tmp = train_list[i * batch_val:(i + 1) * batch_val if (i + 1) < cnt else n_sam]
            imgData = []
            for filename in file_tmp:
                imgData_tmp = np.load(train_path + filename)
                imgData.append(imgData_tmp)
            imgData = np.array(imgData)
            inputData = torch.tensor(imgData).float().cuda()
            with torch.no_grad():
                predict ,weight = model(inputData) #b,c,h,w--b,4,1,1
                loss = criterion(predict, inputData) + 0.01*torch.sum(weight)/img.size()[0]# 对权重实施l1正则化，并除以批量数
                weight = weight.squeeze() #B*C
                weight_ = weight.detach().cpu().numpy()
                weight_batch[i * batch_val:(i + 1) * batch_val if (i + 1) < cnt else n_sam] = weight_
                trainLossTotal += loss.item()
        print('train epoch:', epoch, ': ', trainLossTotal)
        channel_weight_list.append(weight_batch)
        loss_list.append(trainLossTotal)

        scheduler.step(trainLossTotal)
        print('\n')
        mean_weight = np.mean(weight_batch, axis=0)
        band_indx = np.argsort(mean_weight)[::-1][:select_num]
        print('train epoch:', epoch, ', select best bands:',band_indx)
        print('\n')
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), model_path + str(epoch) + '.pkl')

        model.eval()
        weight_batch2 = np.zeros((n_sam2, bands))
        batch_val = mBatchSize * 2
        cnt = math.ceil(n_sam2 / batch_val)
        testLossTotal =0
        for i in range(cnt):
            file_tmp = test_list[i * batch_val:(i + 1) * batch_val if (i + 1) < cnt else n_sam2]
            imgData = []
            for filename in file_tmp:
                imgData_tmp = np.load(test_path + filename)
                imgData.append(imgData_tmp)
            imgData = np.array(imgData)
            inputData = torch.tensor(imgData).float().cuda()
            with torch.no_grad():
                predict, weight = model(inputData)  # b,c,h,w--b,4,1,1
                loss = criterion(predict, inputData) + 0.01 * torch.sum(weight) / img.size()[0]  # 对权重实施l1正则化，并除以批量数
                weight = weight.squeeze()  # B*C 奇怪，为啥之前没报错呢？？
                weight_ = weight.detach().cpu().numpy()
                weight_batch2[i * batch_val:(i + 1) * batch_val if (i + 1) < cnt else n_sam] = weight_
                testLossTotal += loss.item()
        print('test epoch:', epoch, ': ', testLossTotal)
        test_list.append(testLossTotal)
        channel_weight_list_test.append(weight_batch2)

        print('\n')
        mean_weight = np.mean(weight_batch2, axis=0)
        band_indx = np.argsort(mean_weight)[::-1][:select_num*3]
        print('test epoch:', epoch, ', select best bands:', band_indx)

    channel_weight_list = np.array(channel_weight_list)
    np.save('channel_weight_list.npy',channel_weight_list)
    channel_weight_list_test = np.array(channel_weight_list_test)
    np.save('channel_weight_list_test.npy', channel_weight_list_test)