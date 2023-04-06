from utils.dictNew import *
from skimage import io
import numpy as np
label_data_dir = '/home/cjl/dataset/label/'

for file in testfile:
    print(file)
    imgLabel = io.imread(label_data_dir + file + '.png')
    for i in range(30):
        label_index = np.where(imgLabel == i)
        # print(i)
        print(i," :",len(label_index[0]))