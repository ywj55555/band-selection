import numpy as np
import numpy.linalg as lg
import math
from scipy.stats import entropy
from tqdm import tqdm
import random
# kl散度
def KL_div(p, q):
    '''
    输入为两图像的灰度直方图
    return sum p_x*log(px/qx)
    '''
    dis = np.sum(p * np.log2(np.true_divide(p, q)))
    return dis

# 计算SID 光谱信息散度
def SID(x,y):
    # KL_div(p, q) + KL_div(q, p)
    p = np.zeros_like(x,dtype=np.float)
    q = np.zeros_like(y,dtype=np.float)
    Sid = 0
    for i in range(len(x)):
        p[i] = x[i]/np.sum(x)
        q[i] = y[i]/np.sum(y)
        # print(p[i],q[i])
    for j in range(len(x)):
        Sid += p[j]*np.log10(p[j]/q[j])+q[j]*np.log10(q[j]/p[j])
    return Sid

def OIF(s,r,bands = None):
    '''图像数据的标准差越大,所包含的信息量也越大 ; 而波段间的相关系数越小,表明各波段图像数据的独立性越高,信息冗余度越小
    Si表示第 i 个波段的标准差，Rij 表示 i j 两波段的相关系数且要取绝对值
    oif = sum(Si)/sum(|Rij|)
    OIF 越大意味着波段间的相关性越小,包含信息量越丰富
    '''
    if bands:
        s = s[bands]
        r = r[bands,:]
        r = r[:,bands]
    sum_s = np.nansum(s)
    sum_r = 0
    for i in range(len(r)-1):
        for j in range(i+1, len(r)):
            sum_r += r[i][j]
    oif = sum_s /sum_r
    return oif

def JointEntropy():
    pass

def MSD():
    '''
    也可以分小块计算 全部cat成一张图（类比即可） 和计算JM距离时生成数据一样的方式
     MSD evaluates the redundancy among the selected bands, that is, the larger the value
    of the MSD is, the less redundancy is contained among the
    selected bands.
    :return:
    '''
    pass
def cal_mean_spectral_divergence(band_subset):
    """
    Spectral Divergence is defined as the symmetrical KL divergence (D_KLS) of two bands probability distribution.
    We use Mean SD (MSD) to quantify the redundancy among a band set.

    B_i and B_j should be a gray histagram.
    SD = D_KL(B_i||B_j) + D_KL(B_j||B_i)
    MSD = 2/n*(n-1) * sum(ID_ij)

    Ref:
    [1]	GONG MAOGUO, ZHANG MINGYANG, YUAN YUAN. Unsupervised Band Selection Based on Evolutionary Multiobjective
    Optimization for Hyperspectral Images [J]. IEEE Transactions on Geoscience and Remote Sensing, 2016, 54(1): 544-57.

    :param band_subset: with shape (samples, n_band) # 没有类别信息，这是基于信息量的评价指标
    :return:
    """
    # n_row, n_column, n_band = band_subset.shape
    # N = n_row * n_column
    # print(band_subset.shape)
    N, n_band = band_subset.shape
    hist = []
    for i in range(n_band):
        hist_, edge_ = np.histogram(band_subset[:, i], 256)
        # print(hist_.shape)
        hist.append(hist_ / N)
    hist = np.asarray(hist)
    # hist = np.array(hist)
    # print(hist.shape)
    # print()
    hist[np.nonzero(hist <= 0)] = 1e-20
    # entropy_lst = entropy(hist.transpose())
    info_div = 0
    # band_subset[np.nonzero(band_subset <= 0)] = 1e-20
    for b_i in range(n_band):
        for b_j in range(n_band):
            # 波段与波段之间的信息散度 和类别无关！
            band_i = hist[b_i].reshape(-1)/np.sum(hist[b_i])
            band_j = hist[b_j].reshape(-1)/np.sum(hist[b_j])
            entr_ij = entropy(band_i, band_j)
            entr_ji = entropy(band_j, band_i)
            entr_sum = entr_ij + entr_ji
            info_div += entr_sum
    msd = info_div * 2 / (n_band * (n_band - 1))
    return msd
def Entropy():
    pass
# ywj Revise
# https://rdrr.io/cran/varSel/src/R/JMdist.R r语言实现
def JM_distance(m,v):
    '''
    :param mean_v: shape:class_num,spec_num ：每个类别各个通道的均值
    :param cov_v: shape:class_num,spec_num,spec_num ： 每个类别各个通道之间的协方差
    :return: JM—distance
    '''
    n = m.shape[0]
    JM_dist = np.zeros((n, n), dtype=np.float64)

    for class1 in range(n):
        for class2 in range(class1,n):# JM矩阵是一个对称矩阵，不用重复计算
            m_dec = m[class1, :] - m[class2, :]
            v_add = v[class1, :, :] + v[class2, :, :]
            # det->取行列式的值 det(M2M1) = det(M2) * det(M1)
            v_result = lg.det(0.5 * v_add) / math.sqrt(lg.det(v[class1, :, :]) * lg.det(v[class2, :, :]))
            # lob 默认e为底 numpy.matmul矩阵相乘
            # 巴氏距离
            # lg.inv-》逆矩阵
            b = np.matmul(np.matmul(m_dec.T, lg.inv(0.5 * v_add)), m_dec) / 8.0 + math.log(v_result) / 2.0
            JM_dist[class1, class2] = 2 - 2 * math.exp(-b)
            JM_dist[class2, class1] = JM_dist[class1, class2]
    return JM_dist

# 计算skin cloth 和其他类别的JM距离的平均值作为选取波段的评价指标 即 尽可能区分 皮肤,衣物和其他 三类，
def get_average_jm_score(m,v,skin, cloth, water, other):
    """
    :param m:
    :param v:
    :param skin: 一个数字
    :param cloth: 类别列表
    :param water: 一个数字
    :param other: 类别列表
    :return:
    可参考：https://github.com/KolesovDmitry/i.jmdist/blob/master/i.jmdist
    """
    n = m.shape[0]
    JM_dist = np.zeros((n, n), dtype=np.float64)

    for class1 in range(n):
        for class2 in range(class1, n):  # JM矩阵是一个对称矩阵，不用重复计算
            m_dec = m[class1, :] - m[class2, :]
            v_add = v[class1, :, :] + v[class2, :, :]
            # det->行列式 det(M2M1) = det(M2) * det(M1)
            two_det_mul = lg.det(v[class1, :, :]) * lg.det(v[class2, :, :])
            if two_det_mul <= 0 or np.isnan(two_det_mul):
                # return 0
                continue
            v_result = lg.det(0.5 * v_add) / math.sqrt(two_det_mul)
            if v_result <= 0 or np.isnan(v_result):
                # return 0
                continue
            # lob 默认e为底 numpy.matmul矩阵相乘
            # 巴氏距离
            b = np.matmul(np.matmul(m_dec.T, lg.inv(0.5 * v_add)), m_dec) / 8.0 + math.log(v_result) / 2.0
            JM_dist[class1, class2] = 2 - 2 * math.exp(-b)
            JM_dist[class2, class1] = JM_dist[class1, class2]
    score = 0
    cnts = 0
    # 求取两两类别之间的JM的平均值
    # code2label = [0, 2, 2, 2, 0, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for clo in cloth:
        for oth in other:
            if not np.isnan(JM_dist[clo][oth]) and JM_dist[clo][oth] != 0:
                score += JM_dist[clo][oth]
                # score += JM_dist[oth][clo]
                cnts += 1
                # print(JM_dist[clo][oth])
                # if JM_dist[clo][oth]==0:
                #     print('error',clo,oth)
                # print(cnts)
        if not np.isnan(JM_dist[clo][skin]) and JM_dist[clo][skin]!=0:
            score += JM_dist[clo][skin]
            # score += JM_dist[skin][clo]
            # print(JM_dist[clo][skin])
            cnts += 1
            # print(cnts)
            # if JM_dist[clo][skin] == 0:
            #     print('error',clo, skin)
        if not np.isnan(JM_dist[clo][water]) and JM_dist[clo][water]!=0:
            score += JM_dist[clo][water]
            cnts += 1

    for oth in other:
        if not np.isnan(JM_dist[skin][oth]) and JM_dist[skin][oth] != 0:
            score += JM_dist[skin][oth]
            # score += JM_dist[oth][skin]
            # print(JM_dist[skin][oth])
            cnts += 1
            # print(cnts)
            # if JM_dist[skin][oth] == 0:
            #     print('error',skin, oth)
        if not np.isnan(JM_dist[water][oth]) and JM_dist[water][oth] != 0:
            score += JM_dist[water][oth]
            cnts += 1
    if not np.isnan(JM_dist[water][skin]) and JM_dist[water][skin] != 0:
        score += JM_dist[water][skin]
        cnts += 1

    if cnts == 0:
        return 0

    return score/cnts

def get_average_spectral_angle_score(m, skin, cloth, water, other):
    """
    :param m: class_nums, spec_nums
    :param skin:
    :param cloth:
    :param water:
    :param other:
    :return: average spectral angle score
    """
    n = m.shape[0]
    SA_dist = np.zeros((n, n), dtype=np.float64)
    for class1 in range(n):
        for class2 in range(n):  # JM矩阵是一个对称矩阵，不用重复计算
            if class1 == class2:
                continue
            val = np.dot(m[class1].T, m[class2]) / (np.linalg.norm(m[class1]) * np.linalg.norm(m[class2]))  # 默认二阶范数
            SA_dist[class1, class2] = np.arccos(val)

    score = 0
    cnts = 0
    # 求取两两类别之间的JM的平均值
    # code2label = [0, 2, 2, 2, 0, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for clo in cloth:
        for oth in other:
            if not np.isnan(SA_dist[clo][oth]) and SA_dist[clo][oth] != 0:
                score += SA_dist[clo][oth]
                # score += JM_dist[oth][clo]
                cnts += 1
                # print(JM_dist[clo][oth])
                # if JM_dist[clo][oth]==0:
                #     print('error',clo,oth)
                # print(cnts)
        if not np.isnan(SA_dist[clo][skin]) and SA_dist[clo][skin] != 0:
            score += SA_dist[clo][skin]
            # score += JM_dist[skin][clo]
            # print(JM_dist[clo][skin])
            cnts += 1
            # print(cnts)
            # if JM_dist[clo][skin] == 0:
            #     print('error',clo, skin)
        if not np.isnan(SA_dist[clo][water]) and SA_dist[clo][water] != 0:
            score += SA_dist[clo][water]
            cnts += 1

    for oth in other:
        if not np.isnan(SA_dist[skin][oth]) and SA_dist[skin][oth] != 0:
            score += SA_dist[skin][oth]
            # score += JM_dist[oth][skin]
            # print(JM_dist[skin][oth])
            cnts += 1
            # print(cnts)
            # if JM_dist[skin][oth] == 0:
            #     print('error',skin, oth)
        if not np.isnan(SA_dist[water][oth]) and SA_dist[water][oth] != 0:
            score += SA_dist[water][oth]
            cnts += 1
    if not np.isnan(SA_dist[water][skin]) and SA_dist[water][skin] != 0:
        score += SA_dist[water][skin]
        cnts += 1

    if cnts == 0:
        return 0
    return score / cnts

# 为什么会求解为0？？128波段全波段可能有问题？
def get_average_jm_score_four_class(m,v):
    '''
    :param mean_v: shape:class_num,spec_num ：每个类别各个通道的均值
    :param cov_v: shape:class_num,spec_num,spec_num ： 每个类别各个通道之间的协方差
    :return: JM—distance
    '''
    n = m.shape[0]
    JM_dist = np.zeros((n, n), dtype=np.float64)
    for class1 in range(n):
        # for class2 in range(n): # 确实是对称的
            # if class1 == class2:
            #     continue
        for class2 in range(class1, n):  # JM矩阵是一个对称矩阵，不用重复计算
            m_dec = m[class1, :] - m[class2, :]
            v_add = v[class1, :, :] + v[class2, :, :]
            # det->行列式 det(M2M1) = det(M2) * det(M1)
            v_result = lg.det(0.5 * v_add) / math.sqrt(lg.det(v[class1, :, :]) * lg.det(v[class2, :, :]))
            if v_result <= 0 or np.isnan(v_result):
                # return 0
                continue
            # lob 默认e为底 numpy.matmul矩阵相乘
            # 巴氏距离
            b = np.matmul(np.matmul(m_dec.T, lg.inv(0.5 * v_add)), m_dec) / 8.0 + math.log(v_result) / 2.0
            JM_dist[class1, class2] = 2 - 2 * math.exp(-b)
            JM_dist[class2, class1] = JM_dist[class1, class2]
    # return JM_dist
    score = 0
    cnts = 0
    for i in range(1, n):  # other太复杂了，忽略other的计算，应该细分类别计算的！！！
        for j in range(i + 1, n):
            if JM_dist[i, j] == 0:
                # print('class:', i, ' and', j, 'jm score is 0')
                continue
            score += JM_dist[i, j]
            cnts += 1
    if cnts == 0:
        return 0
    return score/cnts

def SequentialFeatureSelection(m,v,select_num, cloth,other,skin,water,forward=True):
    '''
    :param m: n,bands 类别数*候选通道数，每个通道的均值
    :param v: n,bands，bands 类别数*候选通道数*候选通道数，每个通道之间的协方差
    :param select_num: 波段选择数量
    :param cloth: 衣服类别
    :param other: 其他类别
    :param skin: 皮肤类别
    :param water: 水体类别
    :return: 波段选择结果,最终JM距离得分
    '''
    candidate = [x for x in range(m.shape[1])]
    Feature_list = []
    final_score = 0
    if forward:
        for i in range(select_num):
            max_score = 0
            selectedBand = -1
            for band in candidate:
                Feature_list.append(band)
                tmp_m = m[:,Feature_list]
                tmp_v = v[:,Feature_list]
                tmp_v = tmp_v[:,:,Feature_list]
                tmp_score = get_average_jm_score(tmp_m, tmp_v,skin, cloth, water, other)
                Feature_list.remove(band)
                if max_score < tmp_score:
                    max_score = tmp_score
                    selectedBand = band
            print('the ',i,' select ,JM score: ',max_score,' selected band: ',selectedBand)
            if i==select_num-1:
                final_score = max_score
            Feature_list.append(selectedBand)
            candidate.remove(selectedBand)
    else:
        Feature_list = [x for x in range(m.shape[1])]
        for i in range(m.shape[1]-select_num):
            max_score = -1
            selectedBand = -1
            for band in Feature_list:
                Feature_list.remove(band)
                tmp_m = m[:,Feature_list]
                tmp_v = v[:,Feature_list]
                tmp_v = tmp_v[:,:,Feature_list]
                tmp_score = get_average_jm_score(tmp_m, tmp_v,skin, cloth, water, other)
                Feature_list.append(band)
                if max_score<tmp_score:
                    max_score = tmp_score
                    selectedBand = band
            print('the ',i,' SBS ,JM score: ',max_score,' removed band: ',selectedBand)
            if i==m.shape[1]-select_num-1:
                final_score = max_score
            Feature_list.remove(selectedBand)
            # candidate.remove(selectedBand)
    return Feature_list,final_score

def SequentialFeatureSelectionFourClass(m,v,select_num,forward=True):
    '''
    :param m: n,bands 类别数*候选通道数，每个通道的均值
    :param v: n,bands，bands 类别数*候选通道数*候选通道数，每个通道之间的协方差
    :param select_num: 波段选择数量
    :param cloth: 衣服类别
    :param other: 其他类别
    :param skin: 皮肤类别
    :return: 波段选择结果,最终JM距离得分
    '''
    candidate = [x for x in range(m.shape[1])]
    Feature_list = []
    final_score = 0
    if forward:
        for i in range(select_num):
            max_score = 0
            selectedBand = -1
            for band in candidate:
                Feature_list.append(band)
                tmp_m = m[:,Feature_list]
                tmp_v = v[:,Feature_list]
                tmp_v = tmp_v[:,:,Feature_list]
                tmp_score = get_average_jm_score_four_class(tmp_m,tmp_v)
                Feature_list.remove(band)
                if max_score < tmp_score:
                    max_score = tmp_score
                    selectedBand = band
            # print('the ',i,' select ,JM score: ',max_score,' selected band: ',selectedBand)
            if i == select_num-1:
                final_score = max_score
            Feature_list.append(selectedBand)
            candidate.remove(selectedBand)
    else:
        Feature_list = [x for x in range(m.shape[1])]
        for i in range(m.shape[1]-select_num):
            max_score = -1
            selectedBand = -1
            for band in Feature_list:
                Feature_list.remove(band)
                tmp_m = m[:,Feature_list]
                tmp_v = v[:,Feature_list]
                tmp_v = tmp_v[:,:,Feature_list]
                tmp_score = get_average_jm_score_four_class(tmp_m,tmp_v)
                Feature_list.append(band)
                if max_score<tmp_score:
                    max_score = tmp_score
                    selectedBand = band
            print('the ',i,' SBS ,JM score: ',max_score,' removed band: ',selectedBand)
            if i==m.shape[1]-select_num-1:
                final_score = max_score
            Feature_list.remove(selectedBand)
            # candidate.remove(selectedBand)
    return Feature_list,final_score

def SFBS(m,v,selected_list,unselected_list,cloth,other,skin):
    # 前后向搜索！！ 如果直接又128删除来寻找的话，复杂度太大了！！试一下！！
    tmp_m = m[:, selected_list]
    tmp_v = v[:, selected_list]
    tmp_v = tmp_v[:, :, selected_list]
    init_score = get_average_jm_score(tmp_m, tmp_v, cloth, other, skin)
    backup_list = [ x for x in selected_list]
    while(True):
        max_score = 0
        removeBand = -1
        for band in selected_list:
            selected_list.remove(band)
            tmp_m = m[:, selected_list]
            tmp_v = v[:, selected_list]
            tmp_v = tmp_v[:, :, selected_list]
            tmp_score = get_average_jm_score(tmp_m, tmp_v, cloth, other, skin)
            selected_list.append(band)
            if max_score < tmp_score: #找到最小值
                max_score = tmp_score
                removeBand = band
        # if max_score==init_score:
        #     break
        selected_list.remove(removeBand)
        unselected_list.append(removeBand)
        selectedBand = -1
        max_score_add = 0
        for band in unselected_list:
            selected_list.append(band)
            tmp_m = m[:, selected_list]
            tmp_v = v[:, selected_list]
            tmp_v = tmp_v[:, :, selected_list]
            tmp_score = get_average_jm_score(tmp_m, tmp_v, cloth, other, skin)
            selected_list.remove(band)
            if max_score_add < tmp_score:
                max_score_add = tmp_score
                selectedBand = band
        selected_list.append(selectedBand)
        if max_score_add<=init_score:
            break
        else:
            backup_list = [x for x in selected_list]
            init_score = max_score_add
            print('score: ',init_score)
            print('selected : ', selected_list)
    return backup_list,init_score

def SFBSFourClass(m,v,selected_list,unselected_list):
    # 前后向结合搜索！！ 如果直接又128删除来寻找的话，复杂度太大了！！试一下！！
    tmp_m = m[:, selected_list]
    tmp_v = v[:, selected_list]
    tmp_v = tmp_v[:, :, selected_list]
    init_score = get_average_jm_score_four_class(tmp_m, tmp_v)
    backup_list = [ x for x in selected_list]
    while(True):
        max_score = 0
        removeBand = -1
        for band in selected_list:
            selected_list.remove(band)
            tmp_m = m[:, selected_list]
            tmp_v = v[:, selected_list]
            tmp_v = tmp_v[:, :, selected_list]
            tmp_score = get_average_jm_score_four_class(tmp_m, tmp_v)
            selected_list.append(band)
            if max_score < tmp_score: #找到最小值
                max_score = tmp_score
                removeBand = band
        # if max_score==init_score:
        #     break
        selected_list.remove(removeBand)
        unselected_list.append(removeBand)
        selectedBand = -1
        max_score_add = 0
        for band in unselected_list:
            selected_list.append(band)
            tmp_m = m[:, selected_list]
            tmp_v = v[:, selected_list]
            tmp_v = tmp_v[:, :, selected_list]
            tmp_score = get_average_jm_score_four_class(tmp_m, tmp_v)
            selected_list.remove(band)
            if max_score_add < tmp_score:
                max_score_add = tmp_score
                selectedBand = band
        selected_list.append(selectedBand)
        if max_score_add<=init_score:
            break
        else:
            backup_list = [x for x in selected_list]
            init_score = max_score_add
            print('score: ',init_score)
            print('selected : ', selected_list)
    return backup_list,init_score

def SBS(m,v, select_num, cloth,other,skin,water,four_class=True):
    """
    :param m: n,bands 类别数*候选通道数，每个通道的均值
    :param v: n,bands，bands 类别数*候选通道数*候选通道数，每个通道之间的协方差
    :param select_num: 波段选择数量
    :param cloth: 衣服类别
    :param other: 其他类别
    :param skin: 皮肤类别
    :param water: 皮肤类别
    :return: 波段选择结果,最终JM距离得分
    """
    # 只计算后向搜索，可能会很慢
    Feature_list = [x for x in range(m.shape[1])]
    final_score = 0
    for i in range(m.shape[1] - select_num):
        max_score = 0
        selectedBand = -1
        for band in Feature_list:
            Feature_list.remove(band)
            tmp_m = m[:, Feature_list]
            tmp_v = v[:, Feature_list]
            tmp_v = tmp_v[:, :, Feature_list]
            if four_class:
                tmp_score = get_average_jm_score_four_class(tmp_m, tmp_v)
            else:
                tmp_score = get_average_jm_score(tmp_m, tmp_v, skin, cloth, water, other)
            Feature_list.append(band)
            if max_score <= tmp_score:
                max_score = tmp_score
                selectedBand = band
        print('the ', i, ' SBS ,JM score: ', max_score, ' removed band: ', selectedBand)
        if i == m.shape[1] - select_num - 1:
            final_score = max_score
        Feature_list.remove(selectedBand)
    return Feature_list, final_score

def RandomSelection(mean_v, cov_v):
    selected_list = []
    final_score = 0
    for _ in tqdm(range(1000)):
        tmp_selected_list = random.sample(range(128),9)
        # selected_list = [ x for x in range(0,128,128//9) if x >0]
        tmp_m = mean_v[:, tmp_selected_list]
        tmp_v = cov_v[:, tmp_selected_list]
        tmp_v = tmp_v[:, :, tmp_selected_list]
        tmp_final_score = get_average_jm_score_four_class(tmp_m, tmp_v)
        if final_score < tmp_final_score:
            final_score = tmp_final_score
            selected_list = tmp_selected_list
    return selected_list, final_score

# 巴氏距离
def B_distance(m,v):
    '''
        :param mean_v: shape:class_num,spec_num ：每个类别各个通道的均值
        :param cov_v: shape:class_num,spec_num,spec_num ： 每个类别各个通道之间的协方差
        :return: B—distance
        '''
    n = m.shape[0]
    B_dist = np.zeros((n, n), dtype=np.float64)
    for class1 in range(n):
        for class2 in range(n):
            m_dec = m[class1, :] - m[class2, :]
            v_add = v[class1, :, :] + v[class2, :, :]
            v_result = lg.det(0.5 * v_add) / math.sqrt(lg.det(v[class1, :, :]) * lg.det(v[class2, :, :]))
            B_dist[class1, class2] = np.matmul(np.matmul(m_dec.T, lg.inv(0.5 * v_add)), m_dec) / 8.0 + math.log(v_result) / 2.0
    return B_dist

# MH距离 也算是JM距离的一部分
# https://github.com/KolesovDmitry/i.jmdist/blob/master/i.jmdist
def MH_distance_matrix(m,v):
    n = m.shape[0]
    MH_dist = np.zeros((n, n), dtype=np.float64)
    for class1 in range(n):
        for class2 in range(n):
            m_dec = m[class1, :] - m[class2, :]
            v_add = v[class1, :, :] + v[class2, :, :]
            MH_dist[class1, class2] = math.sqrt(np.matmul(np.matmul(m_dec.T, lg.inv(0.5 * v_add)), m_dec))
    print(MH_dist.shape)
    return MH_dist

def divergence_matrix(data):
    # data shape is (n, x, y).
    # where n is number of classes. y is the feature of point x. y是通道数量

    n = data.shape[0]
    divergence = np.zeros((n, n), dtype=np.float64)
    x = data.shape[1]
    if len(data.shape) < 3:
        y = 1
        data = data.reshape(n, x, y)
    else:
        y = data.shape[2]
    print(data.shape, n, x, y)
    m = np.zeros((n, y), dtype=np.float64)
    v = np.zeros((n, y, y), dtype=np.float64)

    for classno in range(n):
        m[classno, :] = np.mean(data[classno, :, :], axis=0)
        # print(data[classno, :, :].reshape(y, x).shape)
        # print(data[classno, :, :].shape)
        v[classno, :, :] = np.cov(data[classno, :, :].reshape(y, x), ddof=1)
    for class1 in range(n):
        for class2 in range(n):
            if class1 == class2:
                continue
            m_dec = (m[class1, :] - m[class2, :]).reshape(y, 1)
            print(m_dec.shape)
            # 求逆矩阵
            v1_inv = lg.inv(v[class1, :, :])
            v2_inv = lg.inv(v[class2, :, :])
            print(v1_inv.shape, v2_inv.shape)
            v_dec = v[class1, :, :] - v[class2, :, :]
            v_inv_dec = v1_inv - v2_inv
            part1 = np.matmul(v_dec, -v_inv_dec)
            print(m_dec)
            print(m_dec.T)
            print(v_inv_dec.shape, m_dec.shape, m_dec.T.shape)
            part2 = np.matmul(np.matmul(v_inv_dec, m_dec), m_dec.T)
            print(part1.shape, part2.shape)
            divergence[class1, class2] = np.trace(part1) / 2.0 + np.trace(part2) / 2.0
    return divergence

# 这个离散度的指标好像不太行！！MSA怎么样？？
def get_average_divergence_score_four_class(m,v):
    '''
    :param mean_v: shape:class_num,spec_num ：每个类别各个通道的均值
    :param cov_v: shape:class_num,spec_num,spec_num ： 每个类别各个通道之间的协方差
    :return: divergence value
    '''
    n = m.shape[0]
    spec_num = m.shape[1]
    divergence = np.zeros((n, n), dtype=np.float64)
    for class1 in range(n):
        for class2 in range(n):
            if class1 == class2:
                continue
            m_dec = (m[class1, :] - m[class2, :]).reshape(spec_num, 1)
            # m_dec = (m[class1, :] - m[class2, :]).reshape(1, spec_num)
            # print(m_dec.shape)
            # 求逆矩阵
            v1_inv = lg.inv(v[class1, :, :])
            v2_inv = lg.inv(v[class2, :, :])
            # print(v1_inv.shape, v2_inv.shape)
            v_dec = v[class1, :, :] - v[class2, :, :]
            v_inv_dec = v1_inv - v2_inv
            part1 = np.matmul(v_dec, -v_inv_dec)  # 到底有没有负号？？？应该是有负号！！！md 为啥不加负号不行啊！！
            # print(m_dec)
            # print(m_dec.T)
            # print(v_inv_dec.shape, m_dec.shape, m_dec.T.shape)
            part2 = np.matmul(np.matmul(v_inv_dec, m_dec), m_dec.T)
            # print(part1.shape, part2.shape)
            divergence[class1, class2] = np.trace(part1) / 2.0 + np.trace(part2) / 2.0
    return divergence
    # score = 0
    # cnts = 0
    # for i in range(1, n):  # other太复杂了，忽略other的计算，应该细分类别计算的！！！
    #     for j in range(i + 1, n):
    #         if divergence[i, j] == 0:
    #             # print('class:', i, ' and', j, 'jm score is 0')
    #             continue
    #         score += divergence[i, j]
    #         cnts += 1
    # if cnts == 0:
    #     return 0
    # return score/cnts

if __name__ == '__main__':
    npypath = '../trainData/'
    select_num = 9
    mean_v = np.load(npypath + 'mean.npy')
    mean_v = mean_v.astype(np.float64)
    # handSelect = [25, 52, 76, 94, 99, 103, 105, 109]
    handSelect = [89, 120, 56, 76, 126, 95, 107, 99, 31]
    print(mean_v.shape)
    cov_v = np.load(npypath + 'cov.npy')
    cov_v = cov_v.astype(np.float64)
    print(cov_v.shape)
    tmp_m = mean_v[:, handSelect]
    tmp_v = cov_v[:, handSelect]
    tmp_v = tmp_v[:, :, handSelect]
    jm = get_average_jm_score_four_class(tmp_m, tmp_v)
    # m = np.random.randint(1, 4, (5, 10))
    # v = np.random.randint(1, 4, (5, 10, 10))
    res = get_average_divergence_score_four_class(tmp_m, tmp_v)
    # jm = get_average_jm_score_four_class(mean_v, cov_v)
    msa = get_average_spectral_angle_score(tmp_m)
    print(res)
    print(jm)
    print(msa)