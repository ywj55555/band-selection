import numpy as np
from data.dictNew import *
import cv2
import os


# other=0, skin=1, cloth=2, plant=3
# no-label pixels: value is set to 255, and will not be counted when calculating
# label         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15
BINARY_CLOTH = [255,  0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0]
BINARY_SKIN  = [255,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0]
BINARY_PLANT = [255,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
MUTI_CLASS   = [255,  0,  0,  0,  3,  3,  1,  2,  2,  2,  0,  0,  0,  0,  0,  0]


def transformLabel(gt, transform_rule):
    ind_list = []
    result = np.zeros(gt.shape, np.int)
    class_nums = len(transform_rule)
    for class_ind in range(class_nums):
        ind_list.append(np.where(gt == class_ind))
    for class_ind in range(class_nums):
        result[ind_list[class_ind]] = transform_rule[class_ind]
    return result


def countPixels(gt, gt_label, predict, predict_label):
    ind_gt = np.where(gt == gt_label)
    ind_correct = np.where((gt == gt_label) & (predict == predict_label))
    if len(ind_gt) > 0:
        tot = len(ind_gt[0])
    else:
        tot = 0
    if len(ind_correct) > 0:
        cnt = len(ind_correct[0])
    else:
        cnt = 0
    return cnt, tot


def evaluateForBinaryClassification(TP, FP, TN, FN):
    if TP + FP > 0:
        precision_rate = TP / (TP + FP)
    else:
        precision_rate = np.nan
    if TP + FN > 0:
        recall_rate = TP / (TP + FN)
    else:
        recall_rate = np.nan
    if TP + FP + TN + FN > 0:
        accuracy = (TP + TN) / (TP + FP + TN + FN)
    else:
        accuracy = np.nan
    if precision_rate + recall_rate > 0:
        F1_score = 2 * precision_rate * recall_rate / (precision_rate + recall_rate)
    else:
        F1_score = np.nan
    return precision_rate, recall_rate, accuracy, F1_score


if __name__ == "__main__":
    label_path = 'E:/BUAA_Spetral_Data/hangzhou/label/'
    test_path = 'E:/output/'
    file_list = trainFile
    for filename in trainFile:
        label_data = cv2.imread(label_path + filename + '.png', cv2.IMREAD_GRAYSCALE)
        #print(label_data.shape)
        test_data = cv2.imread(test_path + filename, cv2.IMREAD_GRAYSCALE)
    '''
    img1 = np.array([[4, 3, 4], [5, 6, 4], [7, 3, 3]])
    img2 = np.array([[3, 3, 4], [3, 6, 6], [4, 5, 3]])
    print(countPixels(img1, 2, img2, 2))
    print(transformLabel(img1, MUTI_CLASS))'''
