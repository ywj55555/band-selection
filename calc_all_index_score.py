from utils.bsUtils import OIF
import numpy as np
import random
from skimage import io
import csv
# from utils.dictNew import *
from utils.bsUtils import JM_distance, B_distance, get_average_jm_score, cal_mean_spectral_divergence, \
    get_average_spectral_angle_score
from utils.load_spectral import envi_loader
from utils.os_helper import mkdir
import time
import os

wavelength = [
449.466,
451.022,
452.598,
454.194,
455.809,
457.443,
459.095,
460.765,
462.452,
464.157,
465.879,
467.618,
469.374,
471.146,
472.935,
474.742,
476.564,
478.404,
480.261,
482.136,
484.027,
485.937,
487.865,
489.811,
491.776,
493.76,
495.763,
497.787,
499.832,
501.898,
503.985,
506.095,
508.228,
510.384,
512.565,
514.771,
517.003,
519.261,
521.547,
523.86,
526.203,
528.576,
530.979,
533.414,
535.882,
538.383,
540.919,
543.49,
546.098,
548.743,
551.426,
554.149,
556.912,
559.718,
562.565,
565.457,
568.393,
571.375,
574.405,
577.482,
580.609,
583.786,
587.015,
590.296,
593.631,
597.021,
600.467,
603.97,
607.532,
611.153,
614.835,
618.578,
622.385,
626.255,
630.19,
634.192,
638.261,
642.399,
646.606,
650.883,
655.232,
659.654,
664.15,
668.72,
673.367,
678.09,
682.891,
687.771,
692.73,
697.77,
702.892,
708.096,
713.384,
718.755,
724.212,
729.755,
735.384,
741.101,
746.906,
752.8,
758.784,
764.857,
771.022,
777.278,
783.626,
790.067,
796.6,
803.228,
809.949,
816.764,
823.675,
830.68,
837.781,
844.978,
852.271,
859.659,
867.144,
874.725,
882.403,
890.176,
898.047,
906.013,
914.076,
922.235,
930.49,
938.84,
947.285,
955.825
]

if __name__ == "__main__":
    # 设置随机种子
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    log = './log/'
    npypath = './trainData/'
    trainDataNpy = './trainData/128bandsFalse_60_mulprocess.npy'
    trainLabelNpy = './trainData/128bandsFalse_60_mulprocess_label.npy'
    class_nums = 4
    savepath = './trainData/'
    mkdir(log)
    spec_num = 128
    # 类别对应可以看labelshower的代码！ D:\ZY2006224YWJ\material-extraction\labelShower
    cloth = [1, 2, 3, 5, 6, 7]
    other = [4]
    other.extend([x for x in range(9, 30)])
    skin = 8
    water = 0
    # 125 以上的就不行了！！
    # 新增两列 最小间隔（分辨率） 最大波段序号（传感器响应问题）
    select_bands_list = {
        # 'handSelected': sorted([25, 52, 76, 94, 99, 103, 105, 109]),
        '本文人工选取': sorted([1, 13, 25, 52, 76, 92, 99, 105, 109]),  # 就要这组了！ 最小间距4个，综合分数看起来应该不低
        # 120  89  56  95 126  76  75  99   3  ---- 71 124  31 107  69  97 100 127  24
        # [3, 56, 75, 76, 89, 95, 99, 120, 126]
        # 额外九个 [24, 31, 69, 71, 97, 100, 107, 124, 127]
        # 相近波段 1, 76, 92, 99,  --- 25, 105,
        '本文嵌入式选取': sorted([120, 89, 56, 95, 126, 76, 75, 99, 3]),  # 有超过125 最小间距1个
        'BS-Nets': sorted([73, 14, 93, 85, 40, 70, 91, 59, 88]),  # 波长间距 最小间距2个
        'SFS': sorted([0, 86, 83, 58, 23, 77, 66, 91, 39]),  # 波长间距 最小间距3个
        'equalIntervalSelected': sorted([14, 28, 42, 56, 70, 84, 98, 112, 126]),
        'SBS': sorted([77, 69, 36, 81, 54, 114, 91, 86, 3]),  # 最小间距5个
        'EAS': sorted([88, 75, 54, 30, 28, 12, 2, 23, 36])  # 最小间距2个
    }
    # 计算JM SAM用31类别
    mean_v = np.load(npypath + 'mean31class.npy')
    print(mean_v.shape)
    cov_v = np.load(npypath + 'cov31class.npy')
    print(cov_v.shape)
    # 计算 MSD OIF 用4分类的数据，因为和类别无关，无所谓
    trainData = np.load(trainDataNpy)
    trainData = trainData.transpose(0, 2, 3, 1)  # b h w c
    # trainLabel = np.load(trainLabelNpy)
    csv2_save_path = log + 'all_index_score_otherHandSelect.csv'
    f2 = open(csv2_save_path, 'w', newline='')
    f2_csv = csv.writer(f2)
    # selected_save_path = log + 'kindsOfMethodSelected.csv'
    # selected_f = open(selected_save_path, 'w', newline='')
    # selected_f_csv = csv.writer(selected_f)
    # for

    category = []
    for k in select_bands_list.keys():
        category.append(str(k))
    # f2_csv.writerow(category)
    print(category)
    # f2_csv.writerow([" ", "equally_spaced", "random_select", "SFS", "SBS", "EAS", "BS-Nets", "hand_select"])

    calc_index_data = np.empty([0, spec_num], dtype=float)
    calc_index_data = np.append(calc_index_data, trainData[:, 5, 5, :], axis=0)
    print(calc_index_data.shape)
    # calc oif
    std = np.std(calc_index_data, axis=0, ddof=1)
    correlation = np.corrcoef(calc_index_data.transpose(1, 0))  # 这些操作不会改变原始数据 calc_index_data
    # time_start = time.time()
    oif_score = []
    for select_bands in select_bands_list.values():
        # print(select_bands)
        tmp_score = OIF(std, correlation, select_bands)
        oif_score.append(round(tmp_score, 3))
    # finalScore = ['final OIF-Score: ']
    # finalScore.extend(oif_score)
    # f2_csv.writerow(finalScore)
    oif_score_nor = np.array(oif_score.copy(), dtype=np.float32)
    oif_score_nor = oif_score_nor / np.max(oif_score_nor)
    print('oif:', oif_score)

    # calc msd
    msd_score = []
    for select_bands in select_bands_list.values():
        tmp_calc_msd_data = calc_index_data[:, select_bands]
        # print(tmp_calc_msd_data.shape)
        tmp_msd_score = cal_mean_spectral_divergence(tmp_calc_msd_data)
        msd_score.append(round(tmp_msd_score, 3))
    # finalScore = ['final msd-Score: ']
    # finalScore.extend(msd_score)
    # f2_csv.writerow(finalScore)
    msd_score_nor = np.array(msd_score.copy(), dtype=np.float32)
    msd_score_nor = msd_score_nor / np.max(msd_score_nor)
    print('msd:', msd_score)

    # calc jm
    jm_score = []
    for select_bands in select_bands_list.values():
        tmp_m = mean_v[:, select_bands]
        tmp_v = cov_v[:, select_bands]
        tmp_v = tmp_v[:, :, select_bands]
        tmp_jm_score = get_average_jm_score(tmp_m, tmp_v, skin, cloth, water, other)
        jm_score.append(round(tmp_jm_score, 3))
    # finalScore = ['final jm-Score: ']
    # finalScore.extend(jm_score)
    # f2_csv.writerow(finalScore)
    jm_score_nor = np.array(jm_score.copy(), dtype=np.float32)
    jm_score_nor = jm_score_nor / np.max(jm_score_nor)
    print('jm:', jm_score)

    # calc msa
    msa_score = []
    for select_bands in select_bands_list.values():
        tmp_m = mean_v[:, select_bands]
        tmp_msa_score = get_average_spectral_angle_score(tmp_m, skin, cloth, water, other)
        msa_score.append(round(tmp_msa_score, 3))
    # finalScore = ['final msa-Score: ']
    # finalScore.extend(msa_score)
    # f2_csv.writerow(finalScore)
    msa_score_nor = np.array(msa_score.copy(), dtype=np.float32)
    msa_score_nor = msa_score_nor / np.max(msa_score_nor)
    print('msa:', msa_score)

    # 归一化分数
    normalization_score_list = []
    for oif, sd, jm, sa in zip(oif_score_nor, msd_score_nor, jm_score_nor, msa_score_nor):
        sum_score = oif + sd + jm + sa
        normalization_score_list.append(sum_score)

    # cal max band_num 换成nm
    max_band_num = []
    for select_bands in select_bands_list.values():
        max_num = np.max(np.array(select_bands))
        max_band_num.append(str(wavelength[max_num]))

    # cal min band_num 换成nm
    min_interval_band = []
    for select_bands in select_bands_list.values():
        bandsList = [wavelength[x] for x in select_bands]
        last = bandsList[0]
        min_interval = 950
        for now in bandsList[1:]:
            tmp_interval = now - last
            last = now
            if min_interval > tmp_interval:
                min_interval = tmp_interval
        min_interval_band.append(min_interval)
    # excel 应该倒过来保存 好复制！ 保存两位小数 四舍五入 遍历的时候四舍五入！
    index_list = ['方法', 'MSD', 'OIF', 'MJM', 'MSA', '综合得分', '最大波长/nm', '最小间隔/nm']
    f2_csv.writerow(index_list)
    for method, sd, oif, jm, sa, score, maxB, minInter in zip(category, msd_score, oif_score,
                                                              jm_score ,msa_score, normalization_score_list,
                                                              max_band_num, min_interval_band):
        tmp_row = [method, sd, oif, jm, sa, score, maxB, minInter]
        f2_csv.writerow(tmp_row)
    f2.close()
    # selected_f.close()
