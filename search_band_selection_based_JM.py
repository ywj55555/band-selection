from utils.bsUtils import SequentialFeatureSelection, SBS, get_average_jm_score
from utils.os_helper import mkdir
import csv
import time
import numpy as np
import random
from tqdm import tqdm

if __name__ == "__main__":
    seed = 2022
    log = './log/'
    npypath = './trainData/'
    mkdir(log)
    select_num = 9
    cloth = [1, 2, 3, 5, 6, 7]
    other = [4]
    other.extend([x for x in range(9, 30)])
    skin = 8
    water = 0
    # csv2_save_path = log + 'JM_BS-Nets.csv'
    # f2 = open(csv2_save_path, 'w', newline='')
    # f2_csv = csv.writer(f2)
    # # f2_csv.writerow([" "])
    # time_start = time.time()
    # 类别均衡与否不影响啊？？？ 那就换31类别！！
    # mean_v = np.load(npypath+'meanClassEqualization.npy')
    # print(mean_v.shape)
    # cov_v = np.load(npypath+'covClassEqualization.npy')
    # print(cov_v.shape)
    mean_v = np.load(npypath + 'mean31class.npy')
    print(mean_v.shape)
    cov_v = np.load(npypath + 'cov31class.npy')
    print(cov_v.shape)
    # 493nm、556nm、638nm、711nm、724nm、752nm、777 nm、790nm、816 nm 减11
    handSelect = [25, 52, 76, 94, 99, 103, 105, 109]  # 这个手选的有点近了
    # embedding_select
    # 120 89 56 95 126 76 75 99 3
    # 89, 120,  56,  76, 126,  95, 107,  99,  31
    embedding_select = [89, 120,  56,  76, 126,  95, 107,  99,  31]  # 反过去推导手选的！
    embedding_select.sort()
    benets_select = [14,  73,  93,  85,  70,  31, 40,  78, 112]
    benets_select.sort()
    sfs_select = [0, 38, 35, 20, 22, 15, 57, 88, 62]
    sfs_select.sort()
    tmp_m = mean_v[:, handSelect]
    tmp_v = cov_v[:, handSelect]
    tmp_v = tmp_v[:, :, handSelect]
    hand_score = get_average_jm_score(tmp_m, tmp_v,skin, cloth, water, other)
    print('hand jm score:', hand_score)

    tmp_m = mean_v[:, benets_select]
    tmp_v = cov_v[:, benets_select]
    tmp_v = tmp_v[:, :, benets_select]
    # bsnets_score = get_average_jm_score_four_class(tmp_m, tmp_v)
    bsnets_score = get_average_jm_score(tmp_m, tmp_v,skin, cloth, water, other)
    print('bsnets jm score:', bsnets_score)

    tmp_m = mean_v[:, embedding_select]
    tmp_v = cov_v[:, embedding_select]
    tmp_v = tmp_v[:, :, embedding_select]
    embedding_score = get_average_jm_score(tmp_m, tmp_v,skin, cloth, water, other)
    print('embedding jm score:', embedding_score)

    tmp_m = mean_v[:, sfs_select]
    tmp_v = cov_v[:, sfs_select]
    tmp_v = tmp_v[:, :, sfs_select]
    sfs_score = get_average_jm_score(tmp_m, tmp_v,skin, cloth, water, other)
    print('sfs jm score:', sfs_score)

    selected_list1, final_score1 = SequentialFeatureSelection(mean_v, cov_v, select_num, cloth,
                                                              other, skin, water, forward=True)
    print('SFS selected bands:', selected_list1)
    print('SFS jm score:', final_score1)
    # SBS只能用4分类了？
    selected_list2, final_score2 = SBS(mean_v, cov_v, select_num, cloth, other, skin, water, four_class=False)
    print('SBS selected bands:', selected_list2)
    print('SBS jm score:', final_score2)


    # tmp_m = mean_v[:, BSNetsBnads]
    # tmp_v = cov_v[:, BSNetsBnads]
    # tmp_v = tmp_v[:, :, BSNetsBnads]
    # final_score = get_average_jm_score(tmp_m, tmp_v, cloth, other, skin)
    # time_end = time.time()
    # print('cost: ', time_end - time_start, ' s')
    # print('selected result: ',BSNetsBnads)
    # print('final score: ', final_score)
    # f2_csv.writerow(BSNetsBnads)
    # f2_csv.writerow(['final score:',final_score])
    # print(jm)
    # print(np.isnan(jm))
    # print(score)
    # f2.close()