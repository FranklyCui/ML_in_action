import numpy as np
import operator

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

