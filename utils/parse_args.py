#-*- coding: UTF-8 -*-
import argparse   #步骤一

def parse_args():
    """
    :return:进行参数的解析
    """
    parser = argparse.ArgumentParser(description="you should add those parameter")        # 这些参数都有默认值，当调用parser.print_help()或者运行程序时由于参数不正确(此时python解释器其实也是调用了pring_help()方法)时，                                                                     # 会打印这些描述信息，一般只需要传递description参数，如上。
    parser.add_argument('--lr','-l',default=0.0001, type=float,help = "The learning rate")
    parser.add_argument('--batchsize','-b', default=32,type=int,help = "The batchsize of model")# 步骤二，后面的help是我的描述
    parser.add_argument('--epoch','-e', default=300, type=int, help="The epoch of train")
    parser.add_argument('--nora', '-n', default=True, type=bool, help="normalize data")
    parser.add_argument('--model_select', '-m', default=1, type=int, help="normalize data")
    parser.add_argument('--model_pafth','-p', default='./model', type=str, help="The path of save model")
    parser.add_argument('--featureTrans', '-f', default=False, type=bool, help="The path of save model")
    parser.add_argument('--intervalSelect', '-i', default=False, type=bool, help="The path of save model")
    parser.add_argument('--activa', '-a', default="sig", type=str, help="The path of save model")
    parser.add_argument('--per_label_nums_per_img', '-c', default=100, type=int, help="The path of save model")

    args = parser.parse_args()                                       # 步骤三
    return args

def parse_test_args():
    """
    :return:进行参数的解析
    """
    parser = argparse.ArgumentParser(description="you should add those parameter")        # 这些参数都有默认值，当调用parser.print_help()或者运行程序时由于参数不正确(此时python解释器其实也是调用了pring_help()方法)时，                                                                     # 会打印这些描述信息，一般只需要传递description参数，如上。
    parser.add_argument('--FOR_TESTSET','-d',default=0, type=int,help = "The predict data")
    parser.add_argument('--model_select', '-m', default=1, type=int, help="normalize data")
    parser.add_argument('--test_batch','-b', default=3,type=int,help = "The batchsize of predict")# 步骤二，后面的help是我的描述
    args = parser.parse_args()                                       # 步骤三
    return args

if __name__ == '__main__':
    args = parse_args()
    # print(args.addresses)            #直接这么获取即可。
    lr = args.lr
    batchsize = args.batchsize
    epoch = args.epoch
    model_path = args.model_path
    nore = args.nora
    print(lr)
    print(batchsize)
    print(epoch)
    print(model_path)
    if nore:
        print('yes')
    else:
        print('no')
