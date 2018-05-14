#!/usr/bin/env python
# *-* conding: UFT-8 *-*

import numpy as np
import re
import random

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

# 将训练样本集转换为词汇向量（词集模型：不考虑词出现次数）
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
    if type(data_set[0]).__name__ == "list":
        for words_vec in data_set:
            vec = row_of_data_to_vec(words_vec, vacab_list)
            data_set_vecs.append(vec)    
    else:
        vec = row_of_data_to_vec(data_set, vacab_list)
        data_set_vecs.append(vec)    
    return data_set_vecs 

# 词集模型：只考虑词在词汇表中是否出现，不考虑出现次数
def row_of_data_to_vec(words_vec,vacab_list):
     #为每个样本向量初始化词汇向量
    vec = np.zeros(len(vacab_list))
    for word in words_vec:
        if word in vacab_list:
            index = vacab_list.index(word)
            vec[index] = 1
        else:
            print("the word: %s is not in vacabulary!" % word)
    return vec
# 词袋模型：考虑词在词汇表中出现的次数
def bag_row_of_data_to_vec(words_vec, vacab_list):
    """
    考虑词在词汇表中出现的次数；计算每次某词出现与否作为特征时，多次出现之间互相独立
    目标词向量进行计算时，出现几次，则连乘时乘几个
    """
    vec = np.zeros(len(vacab_list))
    for word in words_to_vec:
        if word in vacab_list:
            index = vacab_list.index(word)
            vec[index] += 1
        else:
            print("Bag: the word: %s is not in vacabulary!" % vacab_list)
    return vec

# 训练算法：计算先验概率及条件概率
def train_nb(train_matrix, train_category):
    """
    Des:
        训练朴素贝叶斯算法，计算先验概率及条件概率
    Args：   
        train_matrix -- 训练集, 类型为以样本向量为元素的多维数组
        train_category -- 样本向量的类别标签
    Return：
        pro_0 --> 类别0的先验概率
        pro_1 --> 类别1的先验概率
        pro_0_cond_vec --> 类比0的条件概率向量
        pro_1_cond_vec --> 类别1的条件概率向量
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
    # 类别0/1的样本向量相加的和向量（似然估计）
    #class_1_sum_vec = np.zeros(num_words)  
    #class_0_sum_vec = np.zeros(num_words)  
    
    # 类别0/1的样本向量相加的和向量（贝叶斯估计：拉普拉斯平滑）
    class_1_sum_vec = np.ones(num_words)
    class_0_sum_vec = np.ones(num_words)
    
    # 类别0/1的样本向量中元素为1的个数（似然估计）
    #num_class_1 = 0
    #num_class_0 = 0
    
    #类别0/1的样本向量中元素为1的个数（贝叶斯估计：拉普拉斯平滑）
    num_class_1 = 2  #特征可能取值的个数
    num_class_0 = 2
    
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
    # 计算条件概率向量(为避免小数连乘下溢出，取log）
    pro_1_cond_vec = np.log(class_1_sum_vec / num_class_1)
    pro_0_cond_vec = np.log(class_0_sum_vec / num_class_0)
    
    return pro_0, pro_1, pro_0_cond_vec, pro_1_cond_vec
    
# 分类函数：对输入词向量进行分类
def classify_nb(target_vec, pro_0, pro_1, pro_0_cond_vec, pro_1_cond_vec):
    """
    Des:
        对输入词向量进行分类，返回词向量类别
    Args：
        target_vec -- 目标词向量
        pro_0_cond_vec -- 类别0的条件概率向量
        pro_1_cond_vec -- 类别1的条件概率向量
        pro_0 -- 类别0的先验概率
        pro_1 -- 类别1的先验概率
    Return:
        target_vec_label --> 目标词词向量的类别标签
    思路：
        1. 计算基于目标词向量发生的类别1的条件概率pro(1|target_vec):
            1） 计算类别1发生条件下目标词向量发生的概率pro(target_vec|1)
            2) pro(1|target_vec) = pro(target_vec|1) * pro(1)
        2. 计算基于目标词向量发生的类别0的条件概率；
            同步骤1.
    """
    # 目标词两项点乘条件概率向量，为该目标词向量的条件概率向量
    pro_tar_of_class_1 = np.sum(target_vec * pro_1_cond_vec) + np.log(pro_1)
    pro_tar_of_class_0 = np.sum(target_vec * pro_0_cond_vec) + np.log(pro_0)
    
    if pro_tar_of_class_1 > pro_tar_of_class_0:
        return 1
    else:
        return 0

# 测试naive bayes算法
def testing_nb(target_data):
    """
    Des：
        测试naive bayes算法
    Args：
        None
    Return:
        None
    思路：
        1. 构造数据集；
        2. 构造词汇表；
        3. 依据词汇表将数据集处理成词汇向量矩阵；
        4. 将目标向量处理成词汇向量；
        5. 训练naive bayes算法，求得各类别的先验概率和条件概率向量
        6. 对目标向量进行分类，返回其类别
    """
    # 构造训练集，并将训练集向量化
    my_data, data_labels = load_data_set()
    my_vacab = create_vacab_list(my_data)
    data_vecs = words_to_vec(my_data, my_vacab)
    # 构造目标向量，并向量化
    targe_vec = words_to_vec(target_data, my_vacab)
    # 训练算法
    pro_0, pro_1, pro_cond_vec_0, pro_cond_vec_1 = train_nb(data_vecs, data_labels)
    # 分类
    targe_label = classify_nb(targe_vec, pro_0, pro_1, pro_cond_vec_0, pro_cond_vec_1)
    print("the class label of targe point is: %d" % targe_label)
 
# 切分字符串为词列表
def text_parse(target_str):
    """
    Des:
        将字符串切分成词例表
    Args：
        taget_str -- 目标字符串
    Return：
        target_list --> 目标字符串的词列表
    思路：
        1. 将字符串采用任意字符切分；
        2. 对词列表中元素长度判断，大于2的返回
    """
    str_lis = re.split(r'\W+', target_str)
    target_list = [word.lower() for word in str_lis if len(word) > 2]
    return target_list
    
# 垃圾邮件测试系统
def spam_test():
    """
    Des:
        运用naive bayses算法对邮件分类预测，并计算错误率
    Args：
        None
    Return:
        err_ratio --> 分类错误率
    思路：
        1. 准备数据：
            1.1 遍历每个样本邮件
            1.2 读取每个样本邮件内容
            1.3 对每个样本邮件内容切分为词列表
            1.4 将词列表添加到数据集矩阵中
        2. 处理数据：
            2.1 计算词汇表
            2.2 将数据集转化为词汇向量矩阵
            2.3 选取部分数据集添加入test_data，剩余为train_data
        3. 训练算法：
            计算先验概率和条件概率：
        4. 测试算法
            将测试集样本逐一输入分类函数，判断分类正确性
    """
    # 创建数据集、数据集标签
    data_set = []
    class_labels = []
    
    # 利用计数器，遍历每个spam样本（spam和ham邮件个数相当，便可同时一个循环遍历spam和ham各一个样本邮件）
    for i in range(1, 26):
        # 先遍历spam邮件
        try:   
            fr_spam = open("./input/spam/%d.txt" % i)
            content = fr_spam.read()
        except:
            fr_spam = open("./input/spam/%d.txt" % i, encoding = 'Windows 1252')
            content_spam = fr_spam.read()
        content_list = text_parse(content)
        data_set.append(content_list)
        class_labels.append(1)
        try:
            # 再遍历ham邮件
            fr_ham = open("./input/ham/%d.txt" % i)
            content_ham = fr_ham.read()
        except:
            fr_ham = open("./input/ham/%d.txt" % i, encoding = 'Windows 1252')
            content_ham = fr_ham.read()
        content_ham_list = text_parse(content_ham)
        data_set.append(content_ham_list)
        class_labels.append(0)
    # 创建词汇表
    vacab_list = create_vacab_list(data_set)
    
    # 选取部分数据集添加入test_data，剩余为train_data
    train_set = []
    train_labels = []
    test_set = []
    test_labels = []
    
    test_index_list = [int(index) for index in random.sample(range(1, len(data_set)), 10)]
    #巧妙！运用set()集合，直接对test集合索引取补，得train集合，避免大量运算！！**
    train_index_list = list(set(range(50)) - set(test_index_list))   
    
    # 构建训练集
    for index in train_index_list:
        train_set.append(data_set[index])
        train_labels.append(class_labels[index])
    train_vecs = words_to_vec(train_set, vacab_list)
    pro_0, pro_1, pro_cond_vec_0, pro_cond_vec_1 = train_nb(train_vecs, train_labels)
        
    # 构建测试集
    for index in test_index_list:
        test_set.append(data_set[index])
        test_labels.append(class_labels[index])
    test_vecs = words_to_vec(test_set, vacab_list)
    
    errcnt = 0
    for index in range(len(test_vecs)):
        test_predict_labels = classify_nb(test_vecs[index], pro_0, pro_1, pro_cond_vec_0, pro_cond_vec_1)
        if test_predict_labels != test_labels[index]:
            errcnt += 1
            print("the predict of does not match the real labels!")
    err_ratio = errcnt / len(test_vecs)
    return err_ratio
        
def calc_aver_err(run_num):
    err_sum = 0
    for i in range(run_num):
        err_sum += spam_test()
    err_aver_ratio = err_sum / run_num
    return err_aver_ratio
