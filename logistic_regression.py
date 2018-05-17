#!/usr/bin/env python
# -*- conding:UTF-8 -*-

"""
logigstic regression Model
"""

import numpy as np
import matplotlib.pyplot as plt

# 从文件加载数据，返回样本数据集及类别标签列表
def load_data_set():
    """
    Des:
        从文件加载数据，返回样本数据集及类别标签列表
    Args：
        None
    Return：
        data_set -->
        class_labels -->
    """
    data_set = []
    class_labels = []
    fr = open("input/TestSet.txt")
    text_list = fr.readlines()
    for line in text_list:
        # 对输入的每行数据，去掉首位空白字符，并用空格字符切成列表
        line_list = line.strip().split()
        data_set.append([1, float(line_list[0]), float(line_list[1])])
        class_labels.append(float(line_list[2]))
    return data_set, class_labels

# 定义Sigmoid()函数
def sigmoid(args):
    """
    Des:
        返回输入值的sigmoid值
    Args：
        args -- 输入值，可以为实数，或np.array、np.mat格式
    Return：
        输入值args的sigmoid值
    """
    return 1.0 / (1 + np.exp(-args))

# 批量梯度上升算法
def gradient_ascent(data_set, class_labels):
    """
    Des:
        根据输入样本集和样本列表标签，运用梯度下降法找出最优权重
    Args:
        data_set -- 样本数据集
        class_labels -- 样本类别标签
    Return：
        weight_mat --> 最优权重向量
    思路：
        1. 初始化权重；
        2. 计算梯度；
        3. 更新权重
        4. 返回步骤2，直至达到停止条件
        5. 返回最优权重
    """
    data_mat = np.mat(data_set)
    # 将行向量转换为列向量
    labels_mat = np.mat(class_labels).transpose()
    row_num, col_num = data_mat.shape
    # 构建权重向量（列向量）
    weight_mat = np.ones((col_num, 1))  
    max_cycle_times = 500
    alph = 0.001
    # 迭代
    for cnt in range(max_cycle_times):
        # 求出预测值矩阵
        predict_val_mat = data_mat * weight_mat
        # 映射到(0, 1) 值域内
        predict_lab_mat = sigmoid(predict_val_mat)
        err_mat = labels_mat - predict_lab_mat
        # 求梯度，公式为：gradient_vec = data_mat.tranpose() * [real_labels_vec - predict_labels_vec]
        grad_mat = data_mat.transpose() * err_mat
        # 更新权重，公式为：weight_vec = weight_vec + alph(步长) * gradient_vec
        weight_mat = weight_mat + alph * grad_mat
    return weight_mat
     
# 绘制散点图及回归直线
def plot_best_fit(weight_vec):
    """
    Des:
        利用传入数据集绘制散点图，利用传入权重向量，绘制分类回归直线
    Args：
        weight_vec -- 权重向量
    Return：
        None
    思路：
        1. 数据集绘制散点图
            1.1 建立画图
            1.2 建立子图
            1.3 选出正类、负类对象的x、y轴列表
            1.4 绘制散点图
        2. 利用权重向量绘制分类回归直线
            2.1 构建直线方程
            2.2 构建直线的x、y轴参数列表
            2.3 绘制直线图
    """
    data_mat, label_mat = load_data_set()
    # 类型统一转换，以避免类型不一致导致的.shape维度不一致
    data_mat = np.array(data_mat)
    label_mat = np.array(label_mat)
    weight_vec = np.array(weight_vec)
    
    x_coord_1 = []
    y_coord_1 = []
    x_coord_0 = []
    y_coord_0 = []
    num_point = np.shape(data_mat)[0]
    
    # 将每个样本的第一个特征赋值给x，第二个特征赋值给y
    for i in range(num_point):
        if  int(label_mat[i]) == 1:  # 为避免报错，可强制类型转换
            x_coord_1.append(data_mat[i,1])
            y_coord_1.append(data_mat[i,2])
        else:
            x_coord_0.append(data_mat[i,1])
            y_coord_0.append(data_mat[i,2])
    # 创建画布
    fig = plt.figure()
    # 创建轴域
    ax = fig.add_subplot(111)
    # 绘散点图
    ax.scatter(x_coord_1, y_coord_1, s = 30, c = 'red', marker = 's')
    ax.scatter(x_coord_0, y_coord_0, s = 30, c = 'green')
    # 构建分类直线，分类直线对应于sigmoid函数中z=0的直线，左侧z<0为负类，右侧z>0为正类
    # 其中，z = w0*x0 + w1*x1 + w2*x2，-->> 特征图中分类直线应为：0 = w0*x0 + w1*x1 + w2*x2
    x = np.arange(-5.0, 5.0, 0.1)
    y = -(weight_vec[0] + weight_vec[1] * x) / weight_vec[2]
    ax.plot(x,y)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()


# 随机梯度上升算法(初版，步长不变，样本集顺序循环，执行一遍）
def stoch_grad_ascend(data_set, class_labels):
    """
    Des:
        随机梯度上升算法
    Args：
        data_set -- 样本特征数据集
        class_labels -- 类别标签数据集
    Return:
        None
    思路：
        1. 初始化weight值；
        2. 遍历样本集每个样本点
        3. 对每个样本点，计算其当前梯度grad_current
            3.1 计算当前样本点class_label与predict_label差值，
                公式：diff_i = class_label - predict_label
            3.2 计算grad_current梯度
                公式：grad_current = diff_i * data_set[i]
        4. 更新weight值，公式为：weight = weight + alph * grad_current
    """
    data_set = np.array(data_set)
    class_labels = np.array(class_labels)
    num_row, num_column = np.shape(data_set)
    # 初始化权重向量
    weight = np.zeros(num_column)
    alph = 0.01
    # 遍历每个样本点
    for i in range(num_row):
        predict_label = sigmoid(sum(weight * data_set[i]))
        diff = class_labels[i] - predict_label
        grad_cur_point = diff * data_set[i]
        # 更新权重
        weight = weight + alph * grad_cur_point
    return weight 

# 随机梯度上升算法（改进版：1）步长随迭代进行逐步减小，2）随机选取样本点计算梯度更新权值
def stoch_grad_ascent_improve(data_set, class_labels, max_times = 150):
    """
    Des:
        随机梯度上升算法，改进版：
        1）步长随迭代进行逐步减小，
        2）随机选取样本点计算梯度更新权值
    Args：
        data_set -- 样本特征数据集
        class_labels -- 类别标签集
        max_times -- 计算次数，默认为150
    Return：
        weight
    思路：
        1. 初始化weight值，初始化步长alph值；
        2. 循环计算max_times次：
        3. 对样本集进行遍历，每次遍历时随机选取样本点计算其梯度，更新weight值
            3.1 步长更新，公式：alph = 0.01 + 4.0 / (i + j + 1)
            3.2 随机选取样本点
            3.3 计算样本点grad
            3.4 更新weight值
    """
    data_set = np.array(data_set)
    class_labels = np.array(class_labels)
    num_row, num_column = np.shape(data_set)
    # 初始化weight值
    weight = np.zeros(num_column)
    alph = 0.01
    
    # 绘图：weight随迭代次数变化列表
    # weight_list = []
    
    # 计算max_times次
    for cnt_time in range(max_times):
        # 对样本点做随机抽点，用抽中点的grad更新weight值，直至抽完所有样本点
        index_set = list(range(num_row))
        for cnt_point in range(num_row):
            # 更新步长
            alph = 0.01 + 4 / (cnt_time + cnt_point +1)
            # 随机选取样本点，计算其梯度，更新weight
            rand = np.random.randint(0, len(index_set))
            rand_index = index_set[rand]
            predict_label = sigmoid(sum(weight * data_set[rand_index]))
            diff = class_labels[rand_index] - predict_label
            weight = weight + alph * diff * data_set[rand_index]
            del index_set[rand]
        # weight_list.append(weight)
    return weight

# 绘制特征随迭代次数变化图（自己发挥)
def plot_feature_along_iteration(weight_list, num_iter_times):
    """
    Des:
        绘制x_0/x_1/x_2，这3个特征随这迭代次数变化图
    Args：
        weight_list -- 权值随迭代产生的列表
        num_iter_times -- 迭代次数
    Return：
        None
    思路：
        1. 构建画图. fig 
        2. 构建轴域 ax = fig.add_plot(311)
        3. 绘子图1：
            3.1 构建子图列表
                横轴列表：range(0,迭代次数）
                纵轴列表：x_0,列表类型，元素各位为迭代次数
            3.2 绘制
                ax.plot(x,y)
                plt.xlabel("X0")
                plt.ylabel("迭代次数")
                plt.show()
        4. 绘制子图2
        5. 绘制子图3
    """    
    x_0_list = []
    x_1_list = []
    x_2_list = []
    
    for weight in weight_list:
        x_0_list.append(weight[0])
        x_1_list.append(weight[1])
        x_2_list.append(weight[2])
    axis = range(0, num_iter_times, 1)
    
    fig = plt.figure()
    
    ax = fig.add_subplot(311)
    ax.plot(axis, x_0_list, color = 'red', label = 'feature_0')
    plt.ylabel("X0")
    #plt.xlim((0,200))
    #plt.ylim((0, ))
    
    plt.title("value of weight along iteration")
    plt.legend(loc = "uper right")
    
    ax = fig.add_subplot(312)
    ax.plot(axis, x_1_list, color = 'blue', label = 'feature_1')
    plt.ylabel("X1")
    #plt.xlim((0,200))
    
    ax = fig.add_subplot(313)
    ax.plot(axis, x_2_list)
    plt.ylabel("X2")
    #plt.xlim((0,200))

    plt.show()
 
def cycle_stoch_grad(max_times):
    """
    随机梯度上升算法，顺序循环遍历版
    """   
    weight_list = []
    data_set, labels = load_data_set()
    for i in range(max_times):
        weight = stoch_grad_ascend(data_set, labels)
        weight_list.append(weight)
    return weight_list
    
# 用于测试，减少shell工作量           
def test():
    data, label = load_data_set()
    # weight = stoch_grad_ascend(data, label)
    # weight = stoch_grad_ascent_improve(data, label, 300)
    # weight = gradient_ascent(data, label)
    # plot_best_fit(weight)
    
    weight, weight_list = stoch_grad_ascent_improve(data,label,200)
    
    # print(weight)
    # print(weight_list)
    
    # plot_best_fit(weight)
    
    # weight_list = cycle_stoch_grad(200)    
    
    plot_feature_along_iteration(weight_list, 200)
    
    #plot_best_fit(weight_list[-1])
    
    print(weight_list[-1] == weight)
    print(weight_list[-1])
    print(weight)
 
# Logistic分类器
def classify_vec(targ_vec, weight_vec):
    """
    Des:
        对数几率分类器，对输入样本向量分类，正类返回+1，负类返回0
    Args：
        tar_vec -- 待分类目标样本向量
        weight_vec -- 权重向量
    return:
        label -- 目标样本的预测类别
    思路：
        1. 计算输入样本向量的线性回归值z = tar_vec * weight_vec；
        2. 将z值映射到（0,1）值域，sigmoid(z)
        3. 比较预测值sigmoid(z)是否大于0.5，若大于0.5，返回1，否则，返回0
    """
    targ_vec = np.array(targ_vec)
    weight_vec = np.array(weight_vec)
    predict_label = sigmoid(sum(targ_vec * weight_vec))
    if predic_label > 0.5:
        return 1
    else:
        return 0 
   
# 加载数据，并处理成格式数据  (具有通用性）
def load_horse_data(str):
    """
    Des:
        根据输入地址，加载数据，并处理成格式数据
    Args：
        文件地址
    Return:
        data_set -- 样本特征数据集
        class_labels -- 类别标签数据集
    思路：
        1. 打开文件；
        2. 读取文件置列表，每行文件为一个元素；
        3. 遍历每行文件
        4. 将每行文件去掉首位空格，并以空白字符切开成一个列表；
        5. 将每行文件列表的前len()-1位，添加到为一个新列表，并将该新列表添加值data_set[]
        6. 将每行文件列表的-1位，添加到class_labels列表
        7. 返回data_set, class_labels列表
    """
    fr = open(str)
    lines_list = fr.readlines()
    data_set = []
    class_labels = []
    # 遍历每行文本
    for line in lines_list:
        text_list = line.strip().split()
        data_set.append(text_list[:-1])
        class_labels.append(text_list[-1])
    return data_set, class_labels
  
# 马匹死亡率预测     
def colic_test():
    """
    Des:
        用测试集对死亡率预测算法评估，计算错误率
    Args：
        None
    Return: err_ratio --> 预测错误率
    思路：
        1. 准备数据：数据存放在当前目录子目录./input/中
        2. 分析数据：
            2.1 加载数据；
            2.2 格式化数据；
        3. 训练算法：用data_set训练算法，返回weight
        4. 评估算法：用data_test_set中的每个样本评估算法，求出错误数，并计算错误率
    """     
    data_train_set, class_train_labels = load_horse_data("./input/HorseColicTraining.txt")
    data_test_set, class_test_labels = load_horse_data("input/HorseColicTest.txt")
    # 训练算法
    weight = stoch_grad_ascend(data_train_set, class_train_labels)
    num_test = len(data_test_set)
    # 遍历测试集样本点，对算法预测错误率评估
    for i in range(num_test):
        predict_label = classify_vec(data_test_set[i], weight)
        if predict_label != class_test_labels[i]:
            err_cnt += 1
    err_ratio = float(err_cnt / num_test)
    
