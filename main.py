import numpy as np
import random
import csv

if __name__ == '__main__':
    # pixall = [np.empty([0, 128], dtype=np.float32) for _ in range(3)]
    # a = np.random.randint(1,5, (3, 11, 11, 128))
    # pixall[0] = np.append(pixall[0], a[1, 5, 5:6, :], axis=0)
    # pixall[1] = np.append(pixall[1], a[1, 5, 5:6, :], axis=0)
    # print(pixall)
    # for i in range(5, 5):
    #     print('yyy')
    # a = np.random.randint(1,5, (3, 11, 11, 128))
    # b = a[0, 1, 1, [1,5,7]]
    # c = a[0, 1, 1, [5,1,7]]
    # print(b == c)
    # a = list(range(7))
    # b = random.sample(a, 5)
    # print(a)
    # print(b)
    # calc_index_data = np.random.randint(0, 4, (5, 3))
    # print(calc_index_data)
    # std = np.std(calc_index_data, axis=0, ddof=1)
    # correlation = np.corrcoef(calc_index_data.transpose(1, 0))
    # print(calc_index_data)
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
    select_bands_list = {
        # 'handSelected': sorted([25, 52, 76, 94, 99, 103, 105, 109]),
        '本文人工选取': sorted([1, 13, 25, 52, 76, 92, 99, 105, 109]),  # 就要这组了！ 最小间距4个，综合分数看起来应该不低
        # '本文嵌入式选取': sorted([120, 89, 56, 95, 126, 76, 75, 99, 3, 24, 31, 69, 71, 97, 100, 107, 124, 127]),  # 有超过125 最小间距1个
        '本文嵌入式选取': [120, 89, 56, 95, 126, 76, 75, 99, 3, 24, 31, 69, 71, 97, 100, 107, 124, 127],
        # 有超过125 最小间距1个
        'BS-Nets': sorted([73, 14, 93, 85, 40, 70, 91, 59, 88]),  # 波长间距 最小间距2个
        'SFS': sorted([0, 86, 83, 58, 23, 77, 66, 91, 39]),  # 波长间距 最小间距3个
        'equalIntervalSelected': sorted([14, 28, 42, 56, 70, 84, 98, 112, 126]),
        'SBS': sorted([77, 69, 36, 81, 54, 114, 91, 86, 3]),  # 最小间距5个
        'EAS': sorted([88, 75, 54, 30, 28, 12, 2, 23, 36])  # 最小间距2个
    }
    log = './log/'
    selected_save_path = log + 'kindsOfMethodSelected2.csv'
    selected_f = open(selected_save_path, 'w', newline='')
    selected_f_csv = csv.writer(selected_f)
    selected_f_csv.writerow(['方法', '所选特征波段（经过四舍五入）/ nm'])
    for k, v in select_bands_list.items():
        value = ''
        for nm in v:
            value = value + str(round(wavelength[nm])) + '、'
        tmp_cow = [k, value]
        selected_f_csv.writerow(tmp_cow)
    selected_f.close()
    # a = np.random.randint(0, 4, (5, 10))
    # b = a[4, :] - a[3, :]
    # print(b)
    # print(b.shape)
    # b = b.reshape(10, 1)
    # print(b)
    # print(b.shape)
    # spec_num = 4
    # class_num = 2
    # trainLabel = np.random.randint(0, 4, (9,))
    # trainData = np.random.randint(4, 8, (9, 3, 3, spec_num))
    #
    # print(trainLabel.shape)
    # print(trainData)
    # pixall = [np.empty([0, spec_num], dtype=np.float32) for _ in range(class_num)]  # 类别要注意对应
    # class_nums = []
    # for i in range(class_num):
    #     cnt = np.sum(trainLabel == i)
    #     print(cnt)
    #     class_nums.append(cnt)
    #
    # selected_nums = np.min(np.array(class_nums))  # 每个类别选取这么多个样本
    # print(selected_nums)
    #
    # for i in range(class_num):
    #     i_pos_index = np.where(trainLabel == i)  # trainLabel是一维，所以返回一个一维array
    #     i_pos_index = list(i_pos_index[0])
    #     print(i_pos_index)
    #     print(len(i_pos_index))
    #     # pos_list = list(range(len(i_pos_index)))
    #     # select_pos = random.sample(pos_list, selected_nums)
    #     # select_index = i_pos_index[select_pos]
    #     select_index = random.sample(i_pos_index, selected_nums)
    #     print(select_index)
    #     pixall[i] = np.concatenate((pixall[i], trainData[select_index, 1, 1, :]), axis=0)
    #     print(pixall[i])

    # for order, label in enumerate(trainLabel):  # 0 1 2 3
    #     pixall[label] = np.append(pixall[label], trainData[order, 5, 5:6, :], axis=0)

    # a = [2,4,1]
    # a.sort()
    # print(a)
    # print(a[1, 5, 5:6, ].shape)