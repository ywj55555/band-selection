import numpy as np
import numpy.linalg as lg
import math

# J-M distance 这个不太准确
# def JM_distance(d1, d2):
#     '''
#     input: two distuibution with same dim
#     output: jm
#     '''
#     if len(d1) != len(d2):
#         return -1
#     dis = np.sum((d1 ** 0.5 - d2 ** 0.5) ** 2)
#     return dis

# kl散度
def KL_div(p, q):
    '''
    return sum p_x log(px/qx)
    '''
    dis = np.sum(p * np.log2(np.true_divide(p, q)))
    return dis


# 计算SID 光谱信息散度
def SID(x,y):
    # print(np.sum(x))
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


'''
def JM_distance_matrix(data):
    # data shape is (n, x, y).
    # where n is number of classes. y is the feature of point x.
    n = data.shape[0]
    JM_dist = np.zeros((n, n), dtype=np.float64)
    if len(data.shape) < 2:
        print('Calculate JM Distance Error: The size of InputData.shape is less than 2. ')
        return JM_dist
    x = data.shape[1]
    if len(data.shape) < 3:
        y = 1
        data = data.reshape(n, x, y)
    else:
        y = data.shape[2]
    #print(data.shape)
    m = np.zeros((n, y), dtype=np.float64)
    v = np.zeros((n, y, y), dtype=np.float64)
    for classno in range(n):
        m[classno, :] = np.mean(data[classno, :, :], axis=0)
        #print(data[classno, :, :].reshape(y, x).shape)
        #print(data[classno, :, :].shape)
        v[classno, :, :] = np.cov(data[classno, :, :].reshape(y, x), ddof=1)
    for class1 in range(n):
        for class2 in range(n):
            m_dec = m[class1, :] - m[class2, :]
            v_add = v[class1, :, :] + v[class2, :, :]
            v_result = lg.det(0.5 * v_add) / math.sqrt(lg.det(v[class1, :, :]) * lg.det(v[class2, :, :]))
            b = 0.5 * np.matmul(np.matmul(m_dec.T, lg.inv(0.5 * v_add) ), m_dec) + 0.5 * math.log(v_result)
            JM_dist[class1, class2] = math.sqrt(2 - 2 * math.exp(-b))
    #print(JM_dist)
    return JM_dist
'''
# ywj Revise
# https://rdrr.io/cran/varSel/src/R/JMdist.R r语言实现
def JM_distance_matrix(data):
    # data shape is (n, x, y).
    # where n is number of classes. y is the feature of point x.
    # x就是批量数 样本数 y就是各个通道的值！！特征个数
    n = data.shape[0]
    JM_dist = np.zeros((n, n), dtype=np.float64)
    if len(data.shape) < 2:
        print('Calculate JM Distance Error: The size of InputData.shape is less than 2. ')
        return JM_dist
    x = data.shape[1]
    if len(data.shape) < 3:
        y = 1
        data = data.reshape(n, x, y)
    else:
        y = data.shape[2]
    m = np.zeros((n, y), dtype=np.float64) #均值 每类材质各个波段特征的平均值
    v = np.zeros((n, y, y), dtype=np.float64)#协方差 每类材质各个波段特征之间的关系
    for classno in range(n):
        m[classno, :] = np.mean(data[classno, :, :], axis=0)
        # np.cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None,aweights=None)
        # m:一维或则二维的数组，默认情况下每一行代表一个变量（属性,哪一个通道值），每一列代表一个观测(样本数)
        # v[classno, :, :] = np.cov(data[classno, :, :].reshape(y, x), ddof=1)#不能是reshape吧，应该是transpose!!
        v[classno, :, :] = np.cov(data[classno, :, :].transpose(1, 0), ddof=1)  # 不能是reshape吧，应该是transpose!!
    for class1 in range(n):
        for class2 in range(n):
            m_dec = m[class1, :] - m[class2, :]
            v_add = v[class1, :, :] + v[class2, :, :]
            # det->行列式 det(M2M1) = det(M2) * det(M1)
            v_result = lg.det(0.5 * v_add) / math.sqrt(lg.det(v[class1, :, :]) * lg.det(v[class2, :, :]))
            # lob 默认e为底 numpy.matmul矩阵相乘
            # 巴氏距离
            b = np.matmul(np.matmul(m_dec.T, lg.inv(0.5 * v_add)), m_dec) / 8.0 + math.log(v_result) / 2.0
            JM_dist[class1, class2] = 2 - 2 * math.exp(-b)
    return JM_dist

# 巴氏距离
def B_distance_matrix(data):
    # data shape is (n, x, y).
    # where n is number of classes. y is the feature of point x.
    print(data.shape)
    n = data.shape[0]
    B_dist = np.zeros((n, n), dtype=np.float64)
    if len(data.shape) < 2:
        print('Calculate JM Distance Error: The size of InputData.shape is less than 2. ')
        return B_dist
    x = data.shape[1]
    if len(data.shape) < 3:
        y = 1
        data = data.reshape(n, x, y)
    else:
        y = data.shape[2]
    #print(data.shape)
    m = np.zeros((n, y), dtype=np.float64)
    v = np.zeros((n, y, y), dtype=np.float64)
    for classno in range(n):
        m[classno, :] = np.mean(data[classno, :, :], axis=0)
        # v[classno, :, :] = np.cov(data[classno, :, :].reshape(y, x), ddof=1)
        v[classno, :, :] = np.cov(data[classno, :, :].transpose(y, x), ddof=1) #ywj revise
    for class1 in range(n):
        for class2 in range(n):
            m_dec = m[class1, :] - m[class2, :]
            v_add = v[class1, :, :] + v[class2, :, :]
            v_result = lg.det(0.5 * v_add) / math.sqrt(lg.det(v[class1, :, :]) * lg.det(v[class2, :, :]))
            B_dist[class1, class2] = np.matmul(np.matmul(m_dec.T, lg.inv(0.5 * v_add)), m_dec) / 8.0 + math.log(v_result) / 2.0
    return B_dist

def MH_distance_matrix(data):
    # data shape is (n, x, y).
    # where n is number of classes. y is the feature of point x.
    print(data.shape)
    n = data.shape[0]
    MH_dist = np.zeros((n, n), dtype=np.float64)
    if len(data.shape) < 2:
        print('Calculate JM Distance Error: The size of InputData.shape is less than 2. ')
        return MH_dist
    x = data.shape[1]
    if len(data.shape) < 3:
        y = 1
        data = data.reshape(n, x, y)
    else:
        y = data.shape[2]
    # print(data.shape)
    m = np.zeros((n, y), dtype=np.float64)
    v = np.zeros((n, y, y), dtype=np.float64)
    print(m.shape, v.shape)
    for classno in range(n):
        m[classno, :] = np.mean(data[classno, :, :], axis=0)
        # print(data[classno, :, :].reshape(y, x).shape)
        # print(data[classno, :, :].shape)
        # v[classno, :, :] = np.cov(data[classno, :, :].reshape(y, x), ddof=1)
        v[classno, :, :] = np.cov(data[classno, :, :].transpose(y, x), ddof=1) #ywj revise
    print(m.shape, v.shape)
    for class1 in range(n):
        for class2 in range(n):
            m_dec = m[class1, :] - m[class2, :]
            v_add = v[class1, :, :] + v[class2, :, :]
            MH_dist[class1, class2] = math.sqrt(np.matmul(np.matmul(m_dec.T, lg.inv(0.5 * v_add)), m_dec))
    print(MH_dist.shape)
    return MH_dist

def divergence_matrix(data):
    # data shape is (n, x, y).
    # where n is number of classes. y is the feature of point x.

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