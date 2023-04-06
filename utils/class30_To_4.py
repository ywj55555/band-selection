import numpy as np

def convert_to_one_hot(y, C):
    return np.eye(C,dtype=np.int8)[y.reshape(-1)]
# 皮革是类别4 在此归为其他
code2labelSh = [255,2,2,2,0,2,2,2,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #1:skin 2:cloth 3:hair 0:other

sh_class_num = 30
dst_class_nums = 4
for label in range(1, sh_class_num + 1):
    tmp = np.array(convert_to_one_hot(np.array([class_order for class_order in range(dst_class_nums)]), dst_class_nums)[
                               code2labelSh[label]])

def transformLabel(gt, transform_rule):
    ind_list = []
    result = np.zeros(gt.shape, np.int)
    class_nums = len(transform_rule)
    for class_ind in range(class_nums):
        ind_list.append(np.where(gt == class_ind))
    for class_ind in range(class_nums):
        result[ind_list[class_ind]] = transform_rule[class_ind]
    return result


