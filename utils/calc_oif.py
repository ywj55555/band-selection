from utils.bsUtils import OIF
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

if __name__ == "__main__":
    # 设置随机种子
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    log = './log/'
    npypath = './mean_cov_npy/'
    mkdir(log)
    spec_num = 128
    # mkdir(npypath)
    pixall = np.load(npypath+'calc_msd_data.npz')
    csv2_save_path = log + 'OIF.csv'
    f2 = open(csv2_save_path, 'w', newline='')
    f2_csv = csv.writer(f2)
    # testlist = ['20210521095803026']
    select_bands_list=[[14,28,42,56,70,84,98,112,126],
                       [39,10,92,88,71,121,30,77,58],
                       [0,26,85,77,61,70,91,12,51],
                        [52,77,25,68,87,92,5,37,80],
                        [105,72,0,22,37,29,23,8,58],
                       [71, 118 , 91 , 78 , 72 , 77 ,85 ,121 , 10 ],
                        [1,35,54,61,77,90,107,85,94]]
    calc_oif_data = np.empty([0, spec_num], dtype=float)
    for tmp_pix in pixall.values():
        calc_oif_data = np.append(calc_oif_data,tmp_pix , axis=0)

    std = np.std(calc_oif_data, axis=0 ,ddof=1)
    correlation = np.corrcoef(calc_oif_data.transpose(1, 0))
    time_start = time.time()
    score = []
    for select_bands in select_bands_list:
        tmp_score = OIF(std,correlation,select_bands)
        score.append(tmp_score)
    time_end = time.time()
    print('cost: ',time_end-time_start,' s')
    # print(jm)
    # print(np.isnan(jm))
    f2_csv.writerow([" ","equally_spaced","random_select","SFS","SBS","EAS","BS-Nets","hand_select"])
    finalScore = ['final OIF-Score: ']
    finalScore.extend(score)
    f2_csv.writerow(finalScore)
    print(score)
    f2.close()