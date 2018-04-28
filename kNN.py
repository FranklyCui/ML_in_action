import numpy as np
import operator
from os import listdir

def createDataSet():
    """
    Desc: 
        create the data set and it's labels.
    Args:
        None
    Return:
        data_set -- data's feature make up the data_set
        data_labels -- data's lables corresponding to the data's features        
    """
    data_set = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    data_labels = ["A", "A", "B", "B"]
    
    return data_set, data_labels

#分类输入的样本点
def classify(obj_vec, data_set, data_labels, k):
    """
    Desc:
        classify the inputed object according its near k neighoods,and return its label.
    Args:
        obj_vec -- target object
        data_set -- data's feature
        data_labels -- data's labels
        k -- the number of neighoods
    Return:
        label -- label of obj_vec       
    """
   # 目标点与全部样本点间的欧式距离
    data_set_size = data_set.shape[0]  #训练集样本数量   
    mult_obj_mat = np.tile(obj_vec, (data_set_size, 1)) #生成与训练样本集行数（样本个数）相同的矩阵   
    diff_mat = mult_obj_mat - data_set   #取每个样本集与目标点的各特征的差值（矩阵中的元素）   
    sqr_mat = diff_mat**2 #矩阵元素求平方   
    sum_mat = sqr_mat.sum(axis = 1) #求每个训练样本与目标点特征差的平方和（行求和）   
    distance = sum_mat ** 0.5
   
   
   #找出k个近邻点中的label值
    sort_dist_index = distance.argsort()
   
    laber_dic = {}  #用于存储k个近邻样本点中各标签的数目，key为标签名，value为数量
   
    for iter in range(k):
        vote_label = data_labels[sort_dist_index[iter]]  #讲排序数组中元素取出，作为索引传递给标签数据集
        laber_dic[vote_label] = laber_dic.get(vote_label, 0) + 1  #判断第iter个样本的label是否存在laber_dic中，不存在创立，存在则返回值并加1
        
    laber_list = laber_dic.items()  #字典转成列表
    sort_laber_k = sorted(laber_list, key = operator.itemgetter(1), reverse = True)  #依据列表中元素的第二个索引排序
    
    return sort_laber_k[0][0]

# 解析数据：从文本文件中将数据解析为Numpy数据
def file_to_matrix(file_name):
    """
    思路：
    1. 打开文件
    2. 逐行读取文件
    3. 将文件前3列存到特征矩阵
    4. 将文件最后一列存为label向量
    5. 关闭文件
    6. 返回features矩阵及lable向量
    
    Desc：
        从文本文件解析数据，保存为features矩阵及label向量
    Args：
        文本文件路径及文件名
    Return：
        features_mat -- 特征矩阵
        labels_arr --  标签向量
    
    """
    fr = open(file_name)
    
    file_arr = fr.readlines()
    num_data = len(file_arr)
    features_mat = np.zeros((num_data, 3))
    labels_arr = []
    index = 0  # 样本个数迭代计数
    
    for line in file_arr:
        """
        Des:
            提取file_arr前3列赋给features_mat；
            提取file_arr最后一列赋给label_arr.
        """
        line_ele = line.strip()  #去掉首尾空格
        line_list = line_ele.split('\t') #以制表符切分字符串，返回一个元素为子串的列表
        features_mat[index, : ] = line_list[0:3] #将列表前3列赋值给特征矩阵
        labels_arr.append(int(line_list[-1]))
        
        index += 1
        
    return features_mat, labels_arr

def autoNorm(data_set):
    """ 
    Des:
        对样本数据集做归一化处理；
    Args：
        data_set -- 样本个数据集
    Returns
        data_norm -- 归一化后的数据集矩阵
        range_features -- 跨度向量：最大值与最小值之差
        min_features -- 最小值向量：每个特征的最小值向量
    """
    min_features = data_set.min(0)
    max_features = data_set.max(0)
    
    range_features = max_features - min_features
    num_data = data_set.shape[0]  #样本集行数，即样本个数
    min_features_mat = np.tile(min_features, (num_data,1)) #重复min_features向量以构造一个与数据集同样大小的矩阵
    range_f_mat = np.tile(range_features, (num_data,1))
    diff_mat = data_set - min_features_mat
    
    data_norm = diff_mat / range_f_mat
    
    return data_norm, range_features, min_features
    
def data_class_test(test_ratio, data_set, label_vec, k):
    """"
    Des:
        测试分类函数错误了
    Args:
        test_ratio -- 测试样本率
    Returns：
        err_ratio -- 错误率
    思路：
    1. 定义一个测试样本集；
    2. 将测试样本集及其标签逐一输入classify()函数；
    3. 定义一个错误计数器，如预测标签与真实标签不一致则+1；
    4. 返回错误率 = 错误计数器 / 测试样本数    
    """
    num_data = data_set.shape[0]
    err_num = 0
    num_test = int(num_data * test_ratio)
    for index in np.arange(num_test):
        label_sam_test = classify(data_set[index, :], data_set[num_test:, :], label_vec[num_test:], k)
        if label_sam_test != label_vec[index]:
            err_num += 1
            print("the classifier came back the err answer with: %f, the real label is %f" % (label_sam_test, label_vec[index]))
    err_ratio = err_num / num_data
    
    print("the err rate is: %f." % err_ratio)
    return err_ratio

#预测函数
def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    ff_miles = float(input('frequent flier miles earned per year: '))
    percent_tats = float(input("percentage of time spent playing video games: "))
    ice_cream = float(input('liters of ice cream consumed per year: '))
    
    raw_data_mat, label_vec = file_to_matrix("datingTestSet2.txt")
    norm_data_mat, range_vec, min_vec = autoNorm(raw_data_mat)
    targe_vec = np.array([ff_miles, percent_tats, ice_cream]) #构造目标样本点
    
    classifier_result = classify(targe_vec, norm_data_mat, label_vec, 3)
    
    print("测试：分类结果为： %f" % classifier_result)
    print("You will problably like this person: %s" % result_list[classifier_result - 1])
    
# 将图像转化为特征向量
def imag_to_vector(filename):
    return_vect = []
    fr = open(filename)
    file_list = fr.readlines()
       
    for i in range(len(file_list)):
        file_str = file_list[i].strip('\n')
        return_vect.extend(file_str)
        #return_vect.split('\n')
    return return_vect

# 手写数字分类器，并打印错误率
def hand_writing_classify():
    """
    Des:
        手写数字分类器，并打印错误率
    Args:
        None
    Return：
        None
    思路：
        1. 导入训练样本集：特征矩阵及标签向量
            1）获取目录中文件名列表
            2）对文件名逐一进行处理成imag_to_vector图样转化函数形参形式
            3）逐一处理文件imag为矩阵
            4）获取标签向量
        2. 导入测试样本集：特征矩阵及标签
            同Step1
        3. 运用classify()分类函数循环对每个测试样本点分类
            1）将测试样本点与训练样本集、标签向量传递给classify()函数进行分类
            2）判断分类标签与原标签是否一致，不一致则+1
        4. 统计错误并输出错误率
    """ 
    # 导入训练样本集
    training_file_list = listdir("/home/cui/MachineLearninginAction/MachineLearning-1/input/2.KNN/trainingDigits/")
    num_training_file = len(training_file_list)
    hand_write_labels = []
    data_f_mat = np.zeros((num_training_file, 1024))
    
    #遍历每个训练样本点
    for i in range(num_training_file):
        #file_list = training_file_list[i].split('.')  #去掉'.txt'等文件后缀名
        #training_file_list[i] = file_list[0]
        
        hand_write_labels.append(training_file_list[i][0]) #文件名首字母为该文件label
 
        data_f_mat[i, :] = imag_to_vector("/home/cui/MachineLearninginAction/MachineLearning-1/input/2.KNN/trainingDigits/"+training_file_list[i])  #构造特征集特征矩阵
        
        #print("the label is: %s; " % hand_write_labels[i])
        #print("the file name is: %s;" % training_file_list[i])
        
        
    #导入测试样本集
    test_file_list = listdir("/home/cui/MachineLearninginAction/MachineLearning-1/input/2.KNN/testDigits/")
    num_test_file = len(test_file_list)
    test_labels = []
    test_mat = np.zeros((num_test_file, 1024))
    pre_labels = [] #预测分类
    err_count = 0
    
    #遍历每个测试样本点ls 
    for i in range(num_test_file):
        
        
        #file_list = test_file_list[i].split('.')
        #test_file_list[i] = file_list[0]
        
        test_labels.append(test_file_list[i][0])
        test_mat[i, :] = imag_to_vector("/home/cui/MachineLearninginAction/MachineLearning-1/input/2.KNN/testDigits/" + test_file_list[i])
        
        #对测试样本点进行分类预测
        lab = classify(test_mat[i,:], data_f_mat, hand_write_labels, 1)
        
        print("预测标签为：" + lab)
        print("真实标签为： " + test_file_list[i])
       
        pre_labels.append(lab)

        if pre_labels == test_labels[0]:
            err_count += 1
        
    err_rate = err_count / num_test_file  #测试集错误率
    
    print("本次测试共错误%d次，错误率为%f" % (err_count, err_rate))
    
    return err_rate
        
  
