import numpy as np
import random
from copy import deepcopy
from tqdm import tqdm
from utils.bsUtils import get_average_jm_score

# 参考代码 https://blog.csdn.net/weixin_43385826/article/details/118412345

npypath = './trainData/'
mean_v = np.load(npypath + 'mean31class.npy')
print(mean_v.shape)
cov_v = np.load(npypath + 'cov31class.npy')
print(cov_v.shape)
corrcoef = np.load(npypath + 'corrcoef.npy')
print(corrcoef.shape)
mean_R = np.nanmean(corrcoef, axis=0)  # 取所有类别相关系数的平均数 也需要换成31类别的！实际上差不多！
mean_R_recip = 1 / mean_R
twoBandJMScore = np.load(npypath + 'twoBandJMScore.npy')
proportion = [0.5,0.5]  # 两种启发信息所占比重，相关系数和两波段组合的JM距离，可以写进论文中
#  mean_R_recip 和 twoBandJMScore 都没有使用
select_num = 9
allband_num = 128
# cloth = [0, 1, 2, 4, 5, 6]
# other = [3]
# other.extend([x for x in range(8, 29)])
# skin = 7
cloth = [1, 2, 3, 5, 6, 7]
other = [4]
other.extend([x for x in range(9, 30)])
skin = 8
water = 0

random.seed(2022)
np.random.seed(2022)

def pow_mat(mat,p):
    tmp = deepcopy(mat)
    if p==1:
        return tmp
    j = 1
    while j<p:
        j = j<<1
        tmp *= tmp
        if p-j==1:
            tmp *= mat
            break
    return tmp

#轮盘赌选择实现
def choice(rs):
    p = random.random()
    i = 0
    while p>0:
        p -= rs[i]
        i += 1
    return i-1

#蚂蚁决策 后续再改成和当前已选取波段有关的决策 类似SFS
def decision(band_now,jk,alp,beta):
    global bands_phernomones
    tmp_heuristic = [] #启发信息
    tmp_pheno = [] #信息素浓度
    # 遍历下一个可以到达的点 换成波段选择的话，就是遍历下一个可选的波段号
    for item in jk:
        # 实际上求了nanmean后不存在nan了
        if np.isnan(mean_R[band_now][item]) or mean_R[band_now][item]>0.9:
            continue
        # 试一下SFS思想 即 用目前已选波段计算启发信息
        tmp_heuristic.append(mean_R_recip[band_now][item] * proportion[0] + twoBandJMScore[band_now][item]* proportion[1]/2)
        #启发式信息 现计算波段组合的话可能会很慢 直接用相关矩阵和两波段组合的JM距离矩阵作为启发信息
        tmp_pheno.append(bands_phernomones[band_now][item]) #信息素
    tmp_heuristic = np.array(tmp_heuristic)
    tmp_pheno = np.array(tmp_pheno)
    tmp_heuristic = pow_mat(tmp_heuristic,beta)
    tmp_pheno = pow_mat(tmp_pheno,alp)
    p = tmp_heuristic*tmp_pheno #选取下一个点的概率等于 信息素和启发信息二者的结合
    s = np.sum(p)
    p /= s
    id = choice(p)#按概率大小随机选取
    return jk[id]

def decision_baseJM(band_now,tabu,jk,alp,beta):
    global bands_phernomones
    tmp_heuristic = [] #启发信息
    tmp_pheno = [] #信息素浓度
    # 遍历下一个可以到达的点 换成波段选择的话，就是遍历下一个可选的波段号
    for item in jk:
        # 实际上求了nanmean后不存在nan了
        correlation = False
        for select_item in tabu:  # 已经选择的路径
            if mean_R[select_item][item]>0.9:
                correlation = True
                break
        if correlation:
            continue
        # 试一下SFS思想 即 用目前已选波段计算启发信息
        # tmp_heuristic.append(mean_R_recip[band_now][item] * proportion[0] +
        #                      twoBandJMScore[band_now][item]* proportion[1]/2)
        tmp_m = mean_v[:, tabu]
        tmp_v = cov_v[:, tabu]
        tmp_v = tmp_v[:, :, tabu]
        tmp_score = get_average_jm_score(tmp_m, tmp_v, skin, cloth, water, other)
        # tmp_score = tmp_score/2
        tmp_heuristic.append(tmp_score)
        #启发式信息 现计算波段组合的话可能会很慢 直接用相关矩阵和两波段组合的JM距离矩阵作为启发信息
        tmp_pheno.append(bands_phernomones[band_now][item]) #信息素
    if not tmp_heuristic:
        return -1
    tmp_heuristic = np.array(tmp_heuristic)
    tmp_pheno = np.array(tmp_pheno)
    # 启发因子得分为0-2之间，指数运算后不满足单调性，所以不一定得分越高，概率越大，应先归一化到同一数量级，除以2？
    tmp_heuristic = pow_mat(tmp_heuristic,beta)
    tmp_pheno = pow_mat(tmp_pheno, alp)
    p = tmp_heuristic*tmp_pheno #选取下一个点的概率等于 信息素和启发信息二者的结合
    s = np.sum(p)
    p /= s
    id = choice(p)#按概率大小随机选取
    return jk[id]
#蚂蚁周游 ：走完所有城市，选取指定波段数
def walk_cycle(ant,alpha,beta):
    # global cities_dis
    # while ant['Jk'] != []:
    # 循环选择指定波段数
    while len(ant['tabu']) < select_num:
        # id = decision(ant['tabu'][-1],ant['Jk'],alpha,beta)
        id = decision_baseJM(ant['tabu'][-1],ant['tabu'],ant['Jk'], alpha, beta)
        if id == -1:
            break
        ant['Jk'].remove(id)
        # 这个距离没算回到起点的距离
        # ant['L'] += cities_dis[ant['tabu'][-1],id]
        ant['tabu'].append(id)
    # 波段组合JM得分
    tmp_m = mean_v[:, ant['tabu']]
    tmp_v = cov_v[:, ant['tabu']]
    tmp_v = tmp_v[:, :, ant['tabu']]
    ant['L'] = get_average_jm_score(tmp_m, tmp_v, skin, cloth, water, other)
    # ant['L'] += cities_dis[ant['tabu'][-1], ant['tabu'][0]] #加上回到起点的距离

#重置函数
def reset_ant(ant):
    start = random.randint(0,allband_num-1) #随机放置起点 ,random.randint 是闭区间
    ant['tabu'] = [start]
    ant['Jk'] = [i for i in range(allband_num) if i!=start]
    ant['L'] = 0
#初始化蚂蚁群
ants = []
for i in range(50):
    start = random.randint(0,allband_num - 1)
    tmp = {'tabu':[start],'Jk':[i for i in range(allband_num) if i!=start],'L':0}
    ants.append(tmp)

#参数
Np = 50 #蚂蚁数量
alpha = 1  # 信息素比例因子
# beta = 3
beta = 5  # 启发信息因子
Q = 1
rho = 0.1  # 反映信息素在路径上随时间的蒸发速度（在算法中是每一次迭代之后，信息素丢失的比例）

# 是否使用精英AS
EAS = True
# 是否使用排序AS
ASrank = False
# 是否使用MIN-MAX-AS
MMAS = False
noramal = 10
# 求解迭代次数
G = 200
# 记录每次找到的最优解编码
# cities_order = []
bands_order = []

# 记录最优解的目标函数值
# best = [1e5]
best = [0]
# 蚁周模型
info_c = {}
# 蚁密模型
# info_d = {}
# 蚁量模型
# info_q = {}
# 各波段间信息素矩阵
bands_phernomones = np.array([1 for i in range(allband_num)] * allband_num).reshape(allband_num, allband_num)
while G > 0:
    G -= 1
    perGenMax = 0
    perGen_order = []
    print(200-G,' diedai')
    # 蚂蚁周游 并记录最优路径
    for i in range(50):#每只蚂蚁先后出发，实际上可以多进程并行
        walk_cycle(ants[i], alpha, beta) #改只蚂蚁已经走完所有城市，可以是已经选完指定数量波段
        if ants[i]['L'] > best[-1]:
            # 目前为止，所有迭代次数，最优路径
            best.append(ants[i]['L'])
            bands_order.append(ants[i]['tabu'])
        if ants[i]['L'] > perGenMax:
            # 记录该次迭代的 最优路径的长度 因为是JM得分，所以是越大越好
            perGen_order.append(ants[i]['tabu'])
            perGenMax = ants[i]['L']
    info_c[200 - G] = perGenMax #用于绘制曲线
    print(perGenMax)
    # 信息素挥发
    bands_phernomones = (1 - rho) * bands_phernomones

    # 按照蚁周模型更新信息素
    if EAS:
        for s in range(select_num - 1):
            m, n = perGen_order[-1][s], perGen_order[-1][s + 1]  # 最后一条路径就是最优的！
            bands_phernomones[m][n] += Q * perGenMax / noramal
            bands_phernomones[n][m] += Q * perGenMax / noramal
    else:
        for i in range(50):
            for j in range(select_num-1):
                # m, n = ants[i]['tabu'][j - 1], ants[i]['tabu'][j]
                m, n = ants[i]['tabu'][j], ants[i]['tabu'][j + 1]
                # 蚁量模型
                # cities_phernomones[m][n] += Q/cities_dis[m][n]
                # cities_phernomones[n][m] += Q/cities_dis[m][n]

                # 蚁密模型
                # cities_phernomones[m][n] += Q
                # cities_phernomones[n][m] += Q

                # 蚁周模型
                # cities_phernomones[m][n] += Q / ants[i]['L']
                # cities_phernomones[n][m] += Q / ants[i]['L']
                bands_phernomones[m][n] += Q*ants[i]['L']/noramal
                bands_phernomones[n][m] += Q*ants[i]['L']/noramal
    # 清空蚂蚁禁忌表并随机开始城市 进行第二次迭代
    for i in range(50):
        reset_ant(ants[i])
print('best score: ',best[-1])
print('best bands selected: ',bands_order[-1])

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20,8)

x = list(info_c.keys())
y = list(info_c.values())
plt.title('G=200 Np={} alpha={} beta={} Q={} rho={} min_dist={}'.format(Np,alpha,beta,Q,rho,best[-1]))
plt.plot(x,y)
plt.savefig('./ants_{}_{}_{}_{}_{}_eas_decision_baseJM31class_beta5.png'.format(Np,alpha,beta,Q,rho))
