import os
import shutil
import random
import numpy as np
from skimage import io
from data.dictNew import *
label_path = '/home/cjl/dataset/label/'
filelist = os.listdir(label_path)
print(len(filelist))
print(len(trainFile) + len(testFile))
print(trainFile[0] in testFile)
print(len(trainFile))
waterpath = '/home/cjl/waterDataset/TO_WenJun/label/'
labelpath = '/home/cjl/waterDataset/TO_WenJun/label/'
filelist = os.listdir(waterpath)
imgpath = '/home/cjl/waterDataset/TO_WenJun/'
shpath = '/home/cjl/dataset/envi/'
print('water len: ', len(filelist)) #199
cnt = 0
# print()
imgsize = 673404160
# 选150张
# waterFile = []
shenzhenLabel = '/home/cjl/waterDataset/TO_WenJun/labelShenzhen/'
# for file in filelist:
#
#     if file[-4:]!='.png':
#         continue
#     imgLabel = io.imread(waterpath + file)
#     if np.max(imgLabel) == 2:
#         print(file)
#         shutil.copy(waterpath + file, shenzhenLabel + file)
    # print(np.max(imgLabel))
#     # print(os.path.getsize(imgpath + file[3:-4] + '.img') )
#     # break
#     if not os.path.exists(imgpath + file[3:-4] + '.img'):
#         print(file)
#         shutil.move(waterpath + file, imgpath + file)
#         continue
#
#     if not os.path.exists(imgpath + file[3:-4] + '.hdr'):
#         print(file)
#         shutil.move(waterpath + file, imgpath + file)
#         continue
#     if os.path.getsize(imgpath + file[3:-4] + '.img') != imgsize:
#         print(file, 'not equal imgsize!!')
#         shutil.move(waterpath + file, imgpath + file)
#         continue
#     # shutil.move(waterpath + file, labelpath + file)
#     waterFile.append(file[3:])
#     cnt += 1
# print(cnt)
# # 20210722
# samplewaterFile = random.sample(waterFile, 150)
# sampleSh = random.sample(trainFile, 350)

shPngpath = '/home/cjl/dataset/rgb/'
shLabelpath = '/home/cjl/dataset/label/'
waterLabel = '/home/cjl/waterDataset/TO_WenJun/label/'
randomselectShFile = ['20210714110639802', '20210714130847408']
randomselectWaterFile = ['20220426172559745', '20220424162544512']
dstpngpath = '/home/cjl/spectraldata/water_skin_rgb/'
dstlabelpath = '/home/cjl/spectraldata/trainLabelAddWater/'

# for file in randomselectShFile:
#     shutil.copy(shPngpath + file + '.png', dstpngpath + file + '.png')
#     shutil.copy(shLabelpath + file + '.png', dstlabelpath + file + '.png')

for file in randomselectWaterFile:
    # shutil.copy(shPngpath + file + '.png', dstpngpath + file + '.png')
    shutil.copy(waterLabel + 'rgb' + file + '.png', dstlabelpath + file + '.png')

# alltrainFileWater = shDataAndWater
# shData = trainFile
# waterFile = os.listdir(labelpath)
# waterFile = [x[3:-4] for x in waterFile]
# selectShFile = [x for x in shData if x not in alltrainFileWater]
# randomselectShFile = random.sample(selectShFile, 2)
# print(randomselectShFile)
# for file in randomselectShFile:
#     src = shpath + file[:8] + '/' + file + '.img'
#     dst = '/home/cjl/spectraldata/RiverLakeTrainData/'
#     dst = dst + file + '.img'
#     if not os.path.exists(src):
#         print(file, 'not exit')
#     else:
#         shutil.copy(src, dst)
#         src = src[:-4] + '.hdr'
#         dst = dst[:-4] + '.hdr'
#         shutil.copy(src, dst)
#
# selectWaterFile = [x for x in waterFile if x not in alltrainFileWater]
# randomselectWaterFile = random.sample(selectWaterFile, 2)
# print(randomselectWaterFile)
# for file in randomselectWaterFile:
#     src = imgpath + file + '.img'
#     dst = '/home/cjl/spectraldata/RiverLakeTrainData/'
#     dst = dst + file + '.img'
#     if not os.path.exists(src):
#         print(file, 'not exit')
#     else:
#         shutil.copy(src, dst)
#         src = src[:-4] + '.hdr'
#         dst = dst[:-4] + '.hdr'
#         shutil.copy(src, dst)
# 一个挑两张复制过去！

# for file in testFile:
#     if not os.path.exists(shpath + file[:8] + '/' + file + '.img') :
#         print(file)
#     if not os.path.exists(shpath + file[:8] + '/' + file + '.hdr') :
#         print(file)
#     if os.path.getsize(shpath + file[:8] + '/' + file + '.img') != imgsize:
#         print(file, 'not equal imgsize!!')
#         # shutil.move(waterpath + file, imgpath + file)
#         continue
# # 从trainFile中选350张
# for file in trainFile:
#     if not os.path.exists(shpath + file[:8] + '/' + file + '.img') :
#         print(file)
#     if not os.path.exists(shpath + file[:8] + '/' + file + '.hdr') :
#         print(file)
#     if os.path.getsize(shpath + file[:8] + '/' + file + '.img') != imgsize:
#         print(file, 'not equal imgsize!!')
#         # shutil.move(waterpath + file, imgpath + file)
#         continue
#
# train_file = './trainData/all_train_file.txt'
# # test_file = './test_file.txt'
#
# trainf = open(train_file,'w')
#
# print('all train len : ', len(alltrainFileWater))
# for file in alltrainFileWater:
#     if file[-4:] == '.png':
#         # continue
#         trainf.write('\'' + file[:-4] + '\',' + '\n')
#     else:
#         trainf.write('\'' + file + '\',' + '\n')
# # os.close(trainf)
# # os.close(testf)
# trainf.close()
