import pandas as pd
import numpy as np
import collections
import itertools
from pandas import DataFrame
from sklearn.svm import SVC
from sklearn import datasets
from sklearn import svm
import statistics
import warnings
warnings.filterwarnings("ignore")
dataframe_balance= pd.read_csv("SPECT.csv")

col_list=list(dataframe_balance.columns)
col_list.pop(0)
#print(col_list)

#Stratified K fold Validation, splitting the data on the basis of class first.
df_pos=dataframe_balance[dataframe_balance["Overall_Diagnosis"]==1]
df_neg=dataframe_balance[dataframe_balance["Overall_Diagnosis"]==0]

#splitting both the classes into k folds.
df_pos_splits=np.array_split(df_pos,5)
df_neg_splits=np.array_split(df_neg,5)

Feature_list=[]
max_acc_before=0


#Cross Validation starts here, iterating one of the dataframes and creating test set and training set for each fold.
for i in range(len(df_pos_splits)):
    testing_pos=df_pos_splits[i]
    testing_neg=df_neg_splits[i]
    frames=[testing_pos, testing_neg]
    testing_full=pd.concat(frames)


    if i==0:
        training_pos=df_pos_splits[i+1:]
        training_pos_full=pd.concat(training_pos)
        training_neg=df_neg_splits[i+1:]
        training_neg_full=pd.concat(training_neg)
        frames=[training_pos_full,training_neg_full]
        training_full=pd.concat(frames)
    else:
        training_pos = df_pos_splits[0:i] + df_pos_splits[i + 1:]
        training_pos_full = pd.concat(training_pos)
        training_neg = df_neg_splits[0:i] + df_neg_splits[i + 1:]
        training_neg_full = pd.concat(training_neg)
        frames = [training_pos_full, training_neg_full]
        training_full = pd.concat(frames)

        # Storing the class label for the training set for the respective fold.
    class_label_train = training_full['Overall_Diagnosis']
    # print (class_label_train)

    class_label_test = testing_full['Overall_Diagnosis']
    a = (class_label_train.values)

    dic_feature_acc = {}
    # iTerating feature by feature and appending it to a listof features which is emppty in the beginning and it stores that
    # feature which provides the maximum  average accuracy over k folds.
    for col in col_list:

        Feature_list.append(col)

        # Applying svm over the training set to fit the model and then testing it on the tesing set, along with the test labels.
        curr_feature = training_full[Feature_list]
        curr_test_feature = testing_full[Feature_list]
        clf = SVC(gamma='auto')
        clf.fit(curr_feature, class_label_train)
        avg_acc = clf.score(curr_test_feature, class_label_test)
        # removing the col with which we just calculated the avg accuracy and move for calulating accuracy with the next feature.
        Feature_list.remove(col)
        if col not in dic_feature_acc.keys():
            dic_feature_acc[col] = [avg_acc]
        else:
            dic_feature_acc[col].append(avg_acc)
    dic_avgfold_acc = {}
    max_acc = 0
    feature = ""

    # calculating the average of the k folds accuracy calculated ffor a particular feature. if it is greater than some other features accuracy
    # it is selected as the feature and added to the subset.
    for key, value in dic_feature_acc.items():
        avg = statistics.mean(value)
        # print(avg)

        if avg > max_acc:
            max_acc = avg
            feature = key

    # If the accuracy decreases on adding a feature we stop.
    if max_acc > max_acc_before:
        Feature_list.append(feature)
        col_list.remove(feature)
        max_acc_before = max_acc

    else:
        print("True")
        break

print(Feature_list)