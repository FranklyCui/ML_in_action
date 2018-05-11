#!/usr/bin/env python
# *-* conding: UFT-8 *-*

import numpy as np

# 创建一批实验样本
def load_data_set():
    """
    Des:
        创建一批试验样本，及其类标记
    Args:
        None
    Return:
        posting_list, class_vec
    """
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['mybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec
    
# 创建词汇表，以列表形式返回
def create_vacab_list(data_set):
    """
    Des:
        依据输入数据集，创建词汇表，以列表类型返回
    Args：
        data_set -- 输入样本数据集，列表类型
    Return：
        vacab_list
    """
    vacab_set = set([])
    for vec in data_set:
        vacab_set = vacab_set | set(vec)
    vacab_list = sorted(list(vacab_set))
    return vacab_list

# 将训练样本集转换为词汇向量
def words_to_vec(data_set, vacab_list):
    """
    Des:
        将训练样本每个样本向量，转换为以词汇向量
    Args：
        data_set -- 输入样本数据集，应为列表类型
        vacab_list -- 词汇表，列表类型
    Return：
        data_set_vecs --> 以词汇向量为元素的样本集
    思路：
        1. 遍历每个样本向量，初始化词汇向量
        2. 遍历向量中的每个词条
        3. 判断词条是否存在于vacabu_list词汇表中
        4. 若是，则将该词条对应的词汇向量值置为1
    """
    data_set_vecs = []
    # 遍历样本集的每个样本向量
    for words_vec in data_set:
        # 为每个样本向量初始化词汇向量
        vec = np.zeros(len(vacab_list))
        for word in words_vec:
            if word in vacab_list:
                index = vacab_list.index(word)
                vec[index] = 1
            else:
                print("the word: %s is not in vacabulary!" % word)
        # 将样本向量的词汇向量添加到词汇向量集内
        data_set_vecs.append(vec)    
    return data_set_vecs 


 # 训练算法：计算先验概率及条件概率
def train_nb(train_matrix, train_category):
    """
    Des:
        训练朴素贝叶斯算法，计算先验概率及条件概率
    Args：   
        train_matrix -- 训练集, 类型为以样本向量为元素的多维数组
        train_category -- 样本向量的类别标签
    Return：
        pro_0_cond_vec --> 类比0的条件概率向量
        pro_1_cond_vec --> 类别1的条件概率向量
        pro_0 --> 类别0的先验概率
        pro_1 --> 类别1的先验概率
    思路：
        1. 遍历train_matrix的样本向量
        2. 判断该样本向量的类别
        3. 若为1，
            1）对所有类别为1的样本向量相加，所得向量即为词汇表中单词的个数
            2）对所有类别为1的样本向量的元素求和，即计算样本向量中出现词汇表单词的总数
            3）统计所有类别为1的个数
        4. 若为0：
            同若为1
        5. 计算先验概率和条件概率：
            先验概率：pro_1 = 类别为1的样本向量个数 / 样本向量总数
            条件概率向量：pro_1_cond_vec = 类别为1的样本向量相加 / 类别为1的样本向量元素求和
    """
    
    num_train_vec = len(train_matrix)
    num_words = len(train_matrix[0])
    # 类别0/1的样本向量相加的和向量
    class_1_sum_vec = np.zeros(num_words)
    class_0_sum_vec = np.zeros(num_words)
    # 类别0/1的样本向量中元素为1的个数
    num_class_1 = 0
    num_class_0 = 0
    # 类别0/1的样本向量个数
    cnt_1 = 0
    cnt_0 = 0
    
    # 遍历所有样本向量
    for i in range(num_train_vec):
        # 判断样本向量类别
        if train_category[i] == 1:
            class_1_sum_vec += train_matrix[i]
            num_class_1 += sum(train_matrix[i])
            cnt_1 += 1
        else:
            class_0_sum_vec += train_matrix[i]
            num_class_0 += sum(train_matrix[i])
            cnt_0 += 1
    # 计算先验概率
    pro_1 = cnt_1 / num_train_vec
    pro_0 = cnt_0 / num_train_vec
    # 计算条件概率
    pro_1_cond_vec = class_1_sum_vec / num_class_1
    pro_0_cond_vec = class_0_sum_vec / num_class_0
    
    return pro_0, pro_1, pro_0_cond_vec, pro_1_cond_vec
    
        
        
        
        
