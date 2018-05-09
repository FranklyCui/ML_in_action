#!/usr/bin/env python
#*-*coding:utf-8*-*

import numpy as np
import math
import operator

# 计算输入样本数据集的熵
def calc_Ent(data_set):
        """
        Des:
                计算给定数据集的经验熵，计算公式为：sum(-pk * log2(pk))
        Args:
                data_set -- 训练集，以样本点列表为元素的列表，最后一列为特征标签；
        Return：
                Ent -- 经验熵
        
        思路：
                1. 计算频率pk；
                2. 计算-pk * log2(pk);
                3. 求和，返回值       
        """     
        labels = {}
        num_set = len(data_set)
        Ent = 0
        
        #构造一个以类标记为键，以类标记个数为值的字典
        for fec_vec in data_set:
                label = fec_vec[-1]
                if label in labels.keys():
                        #如果fec_vec的label已存在于labels{}字典中，则label键对应值+1
                        labels[label] += 1  
                else:
                        #否则，将label键添加进labels{}中，并置值为1
                        labels[label] = 1
        
        #计算每个类的频率
        for label in labels:
                pk = float(labels[label]) / num_set
                Ent -= pk * math.log(pk, 2)                     
        return Ent
        
# 创建样本数据集
def cre_data_set():
        data_set = [[1, 1, 'yes'],
                                [1, 1, 'yes'],
                                [1, 0, 'no'],
                                [0, 1, 'no'],
                                [0, 1, 'no']]
        feat_labels = ['no surfacing', 'flippers']
        return data_set, feat_labels

# 切分样本数据集，输入某特征和取值，返回样本集与该特征的值相同的样本子集，并去除该特征
def split_data_set(data_set, axis, value):
        """
        Des:
                以输入特征和值切分样本集，返回与输入特征和值相同的样本子集，并去除该特征
        Args：
                data_set -- 输入样本集
                axis -- 输入特征
                value -- 输入特征的取值
        Return：
                sub_data_set -- 输入样本集中与输入特征和值相同的样本子集
                
        思路：
                1. 取出data_set每一个样本，判断样本特征的值是否与输入值相等；
                2. 若相等，则在该样本点中去掉输入特征后，加入样本子集；
                3. 若不等，继续下一个样本
        """
        sub_data_set = []       
        for fec_vec in data_set:
                if fec_vec[axis] == value:
                        # 字符串切片，去掉axis列特征
                        sub_fec_vec = fec_vec[:axis]
                        sub_fec_vec.extend(fec_vec[axis+1:])
                        sub_data_set.append(sub_fec_vec)
        return sub_data_set

# 返回信息增益最大的特征作为最优特征
def choose_best_feature(data_set):
        """
        Desc:
                计算每个特征的信息增益，选取信息增益最大的特征作为最优划分特征
        Args:
                data_set -- 样本数据集，每个样本点构成一个列表元素，最后一列为label
        Return:
                best_feature -- 信息增益最大的特征
                
        思路：
                1. 计算每个特征的信息增益；
                        1.1 计算当前样本信息熵；
                        1.2 将样本按照当前特征的取值拆分成子样本集；
                        1.3 计算子样本集的信息熵，计算所有子集加权熵；
                        1.4 计算信息增益。
                2. 选取信息增益最大的特征
        """             
        cur_data_Ent = calc_Ent(data_set)
        num_feat = len(data_set[0]) - 1    
        best_feature = -1
        best_info_Gain = 0
        
        # 遍历每个特征
        for i in range(num_feat):
            feat_Ent = 0            
            feature_lis = [feat_vec[i] for feat_vec in data_set]
            # 利用set()函数对list去重
            feature_set = set(feature_lis)  
            # 遍历特征的每个取值
            for value in feature_set:
                # 以特征i、值value切分出样本子集
                sub_data_set = split_data_set(data_set, i, value)
                # 计算子集信息熵
                sub_Ent = calc_Ent(sub_data_set)
                prob = float(len(sub_data_set) / len(data_set))
                feat_Ent += prob * sub_Ent
            # 当前特征的信息增益
            feat_info_Gain = cur_data_Ent - feat_Ent
            if feat_info_Gain > best_feature:
                best_info_Gain = feat_info_Gain
                best_feature = i
        return best_feature
        
# 返回某样本集中数目最多的label
def majority_cnt(class_label):
    """
    Des:
        返回样本集合中数目最多的类
    Args:
        class_label -- 样本集合列表，元素为样本特征列表
    Return：
        muti_label -- 样本集合中数目最多的类
    思路：
        1. 定义一个空字典，key为label，value为数量；
        2. 遍历样本集合，判断每个样本点label是否存在于字典中；
        3. 若不存在，将该label添加进字典，并置value为1；
        4. 若存在， 对该label对应值+1；
        5. 将字典重组为以“键值对”元组为元素的列表；
        6. 以值大小降序对列表排序；
        7. 返回列表中第一个元素的键。   
    """ 
    count_dic = {}
    for label in class_label:
        if label not in count_dic:
            count_dic[label] = 0
        count_dic[label] += 1
    count_list = count_dic.items()
    # 对以“键值对”元素为元素的列表以索引1进行排序
    re_count_list = sorted(count_list, key = operator.itemgetter(1), 
                    reverse = True)
    muti_label = re_count_list[0][0]
    return muti_label
                
# 创建决策树
def create_tree(data_set, feat_labels):
    """
    Des:
        创建决策树
    Args：
        data_set -- 样本集合
        feat_labels -- 特征标签向量，以阐释特征含义的字符串列表
    Return：
        my_tree -- 决策树，存储每个节点的最优特征及对应分支的特征取值
    思路：
        1. 判断当前样本集是否为同一类别，若是，则return类标签
        2. 判断当前属性集是否为空，若是，则return样本集中多数类标签
        3. 选取当前样本集最优特征，取出最优特征的值
        4. 遍历最优特征的值，依据取值将当前样本集切分成不同子集；
        5. 对子集重复步骤1～步骤4。        
    """
    class_labels_list = [item[-1] for item in data_set]
    # 若类标签向量内元素相同，则返回类标签值
    if len(class_labels_list) == class_labels_list.count(class_labels_list[0]):
        
        #**test**
        #print("the class_label_list[0] is: ")
        #print(class_labels_list[0])
        
        return class_labels_list[0]
        
    # 若样本集中属性集为空（仅剩类标签列），则返回当前样本集类标签最多
    if len(data_set[0]) == 1:
        muti_label = (class_labels_list)
       
         #**test**
        #print("test: muti_labe is :", end = "")
        #print(muti_labe)
        
        return muti_label
    # 取最优划分特征及其取值
    best_feat = choose_best_feature(data_set)
    best_feat_value_list = [item[best_feat] for item in data_set]
    # 对最优特征取值去重
    bes_fea_val_set = set(best_feat_value_list)
    # 建立以best_feat_label为“键”的空字典，用于存储不同value对应的分支树结构
    best_feat_label = feat_labels[best_feat]
    my_tree = {best_feat_label:{}}
    
    # 去掉特征标签向量中的最优特征对应的标签
    feat_labels.pop(best_feat)  #list.pop()返回值为被去掉值
    sub_labels = feat_labels[:]
    # 遍历最优特征的值
    for val in bes_fea_val_set:
        sub_data_set = split_data_set(data_set, best_feat, val)
        my_tree[best_feat_label][val] = create_tree(sub_data_set, sub_labels)
    return my_tree
    
      
    
        
