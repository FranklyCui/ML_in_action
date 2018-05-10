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
        data_set -- 输入样本数据集，应为列表类型
    Return：
        vacab_list
    """
    vacab_set = set([])
    for vec in data_set:
        vacab_set = vacab_set | set(vec)
    vacab_list = sorted(list(vacab_set))
    return vacab_list
    
    
