import numpy 
import operator

def createDataSet():
    """
    训练数据集产生函数：返回一个训练数据集，及标记
    """
    group = numpy.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    
    return group, labels



