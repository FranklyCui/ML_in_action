#!/usr/bin/env python
# -*- conding:utr-8 -*-

import numpy as np

# 创建数据集
def load_sim_data():
    """
    Des：
        创建数据集，返回样本集和列表标签集
    Args：
        None
    Return：
        data_set --> 样本集
        class_label -->标签集
    """
    
    data_set = [[1, 2.1], [2,1.1], [1.3, 1.0], [1.0, 1.0], [2.0, 1.0]]
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_set, class_labels


# 根据feature列的value值，对样本类别划分
def stump_classify(data_set, dimen, thresh_val, inequ_sign):
    """
    Des:
        依据输入符号，以输入阀值对样本集某列的值划分，以将样本集分类，返回分类结果
    Args：
        data_set -- 样本集
        dimen -- 样本集某列索引（某特征）
        thresh_val -- 阀值
        ineq_sig -- 输入符号，取值：less_than、greater_than
    Return:
        predict_label_mat --> 预测的分类
    思路：
        1. 将预测分类向量初始化为1：predict_label = np.ones()
        2. 取出data_set[dimen]列，逐一判断value值与threash_value关系
        3. 若根据输入符号，将大于或小于阀值的样本点预测类别置为-1.0
    """
    row_num, column_num = np.shape(data_set)
    predict_label_mat = np.ones((row_num, 1))
    if inequ_sign == "less_than":
        predict_label_mat[data_set[:, dimen] <= thresh_val] = -1.0
    elif inequ_sign == "greater_than":
        predict_label_mat[data_set[:, dimen] >= thresh_val] = -1.0
        
   
    # 判断输入逻辑
    #if ineq_sig == "less_than": 
         # 遍历输入dimen列的value值，并与thresh_value比较
        #for i in range(row_num):
            #if data_set[dimen][i] <= thresh_val:
                #predict_label_mat[i] = -1.0
    #elif ineq_sig == "greater_than":
        #for i in range(row_num):
            #if data_set[dimen][i] > thresh_val:
                #predict_label_mat[i] = -1.0
                
                
    return predict_label_mat
 
  
# 计算决策桩的err_val        
def calc_err(data_mat, label_mat, feat_index, 
            thresh_val, inequ_sign, w_point_mat):
    """
    Des:
        计算以feat_index列为最优特征，以输入thresh值为阀值作为决策桩将data_set划分后的err
    Args：
        data_mat -- 特征数据集
        label_mat --类别标签集
        feat_index -- 最优特征索引
        thresh_val -- 划分阀值
        inequ_sign --选择阀值哪一侧置为负类
        w_point_mat -- 当前样本集样本点的权重向量
    Return:
        err_val --> 当前决策桩的误差
    思路：
        1. 根据输入列、阀值、比较符号，给出样本集的预测类别，predict_label = stump_classify()
        2. 计算预测类别和真是类别的差值向量，
    """
    
    w_point_mat = np.array(w_point_mat)
    
    err_vec = np.ones(len(data_mat))
    predict_label_mat = stump_classify(data_mat, feat_index, thresh_val, inequ_sign)
    
    label_mat = np.array(label_mat)
    predict_label_mat = np.array(predict_label_mat).T
    
    # 测试：
    #print("label_mat is: " + str(label_mat))
    #print("predict_label_mat.T is: " + str(predict_label_mat))
    #print("err_vec is: " + str(err_vec))
    
    # np.array生成式：将两个比较列表中对应元素相等的索引传递给err_vec，并将该索引的值置为0  # 很巧妙！！
    err_vec[label_mat[0,:] == predict_label_mat[0,:]] = 0
    # 计算误差
    err_val = np.dot(w_point_mat, err_vec)
    return err_val
        

# 建立决策树桩
def build_stump(data_set, class_label, w_point_mat):
    """
    Des:
        依据输入的样本特征集、样本类别标签集以及样本点的分布权重，找出错误率最低的决策树桩，并返回
    Args：
        data_set -- 样本特征数据集
        class_label -- 样本类别标签集
        w_point_mat -- 样本点在样本集上的权重
    Return:
        best_stump --> 最优决策桩
        err_best_stump --> 最优决策桩误差， 公式：err = sum(w* I(pridict != real_label))
    思路：
        1. 遍历特征，以找出最优划分特征；
            1.1 遍历特征所有取值（标称型数据类型） 或 在特征取值的最大值与最小值之间，取n个点遍历划分
            1.2 计算当前划分的err
            1.3 取出最优划分特征的最优划分阀值，作为best划分树桩
    """
    data_mat = np.mat(data_set)
    label_mat = np.mat(class_label)
    num_step = 10
    row_num, column_num = np.shape(data_set)
    
    best_stump = {}
    err_best_stump = np.inf
    best_feat = None
    best_thresh_value = None
    best_inequ_sign = None
    
    # 遍历所有特征
    for i in range(column_num):
        # 对与数值型特征，依据步数确定遍历的阀值点
        min_column = data_mat[:,i].min()
        max_column = data_mat[:,i].max()
        step_width = (max_column - min_column) / num_step
        
        #print("min is :" + str(min_column))
        #print("max is :" + str(max_column))
        #print("step_width is : " + str(step_width))
        
        # 遍历选定的阀值点
        for j in np.arange(min_column, max_column, step_width):
            # **!!** 理解如下：确定划分阀值后，因不确定应该将阀值左、右两侧的点，划分为正类、负类，故都要计算err试一下
            for inequ_sign in ["less_than", "greater_than"]:
                # **!!重点理解!!**
                err_cur = calc_err(data_mat, label_mat, i, j, inequ_sign, w_point_mat)   # 根据输入特征的阀值进行分类，计算err值
                
                # 打印出当前决策桩的特征、阀值、判断规则以及误差
                #print("split: dim is: %d; thresh_val is: %.2f; \
                    #inequ_sign is: %s; err is: %.3f" % (i, j, inequ_sign, err_cur))
                
                if err_cur < err_best_stump:
                    err_best_stump = err_cur
                    best_feat = i
                    best_thresh_value = j
                    best_inequ_sign = inequ_sign
                    
                    #print("updated: err_best_stump is: %.3f; best_feat is: %d; best_thresh_value is: %0.2f" % \
                        #(err_best_stump, best_feat, best_thresh_value))
                    
    best_stump["dimen"] = best_feat
    best_stump["inequ_sign"] = best_inequ_sign
    best_stump["best_thresh_value"] = best_thresh_value
    return best_stump, err_best_stump
  

# adaboost算法
def adaBoost_train(data_set, class_labels, max_iter_times = 40):
    """
    Des:
        实现adaBoost算法，公式：f(x) = alph0 * f0(x) + alph1 * f1(x) + ...+alphm * fm(x)
    Args:
        data_set -- 样本特征集
        class_labels --类别标签集
        max_iter_times -- 最大迭代次数，默认值40
    Return：
        classifier_vec -- 分类器向量
        alph_vec -- 分类器系数向量
        公式：classifier = sum(alph_i * sub_classifier_i), i = 1, 2, ..., N
                                        sub_classifer = best_stump  # 决策桩
                                        alph_i = 1/2 * ln((1-err)/err)
    思路：
        1. 思想：前向分步法，逐步计算及基分类器
        2. 初始化：classifier_0, 样本权值向量w_point_0'
        3. 求得当前分步的最优分类器classifier_i及分类误差err_i
        4. 求得该最优分类器的系数alph_i
        5. 返回，最优分类器向量以及系数向量
    """
    data_set = np.array(data_set)
    class_labels = np.array(class_labels)

    classifier_vec = []
    alph_vec = []
    err_classifier_vec = []
    agg_predict_labels = np.zeros(len(data_set))
    w_point_vec = np.ones(len(data_set)) / len(data_set)
    
    for i in range(max_iter_times):
        classifier, err_classifier = build_stump(data_set, class_labels, w_point_vec)
        alph = 0.5 * np.log((1 - err_classifier) / err_classifier)
        classifier_vec.append(classifier)
        err_classifier_vec.append(err_classifier)
        alph_vec.append(alph)
        
        # 计算当前基分类器对样本集的预测值
        predict_labels = stump_classify(data_set, classifier["dimen"], classifier["best_thresh_value"], classifier["inequ_sign"])
        predict_labels = np.array(predict_labels.T)[0]  # predict_label 为np.matrix类型，先转为行向量，再提取为
        
        # 测试
        #print("agg_predict_labels is：" + str(agg_predict_labels) + "; and the shape is:" + str(agg_predict_labels.shape))
        #print("predict_labels is: " + str(predict_labels) + "; and the shape is: " + str(predict_labels.shape))
        
        # 记录总分类器的预测结果
        agg_predict_labels += alph * predict_labels
        
        # 更新样本集权重分布，为下一次计算做准备
        diff_vec = np.multiply(class_labels, predict_labels)  # 根据预判对错，确定是exp(alph)还是exp(-alph)
        w_point_vec = w_point_vec * (np.exp(-alph * diff_vec))
        w_point_vec = w_point_vec / np.sum(w_point_vec)
        
        # 判断是否满足终止条件
        # agg_err_vec  = np.zeros(len(data_set)
        # agg_err_vec[agg_predict_labels != class_labels] =  1.0   # 尝试：看下与下行代码结果有和不同
        agg_err_vec = np.multiply(agg_predict_labels != class_labels, np.ones(len(data_set)))
        err_ratio = np.sum(agg_err_vec) / len(data_set)
        if err_ratio == 0:
            print("err_ratio is: zero!")
            break
    return classifier_vec, alph_vec, agg_predict_labels, agg_err_vec
    
    
# 测试函数
def test():
    data_set, class_labels = load_sim_data()     
    #print(data_set)
    #print(class_labels)
    
    #w_point_mat = np.ones(len(data_set)) / len(data_set)
    
    #best_stump, err_best_stump = build_stump(data_set, class_labels, w_point_mat)
    
    classifier_vec, alph_vec,agg_predict_labels, agg_err_vec = adaBoost_train(data_set, class_labels)
    print("classifier_vec is: " + str(classifier_vec) + "; \nalph_vec is: " + str(alph_vec) + "; \nagg_err_vec is: " + str(agg_err_vec))
  
    
if __name__ == "main":
    test()
