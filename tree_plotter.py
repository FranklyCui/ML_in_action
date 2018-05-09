#!/usr/bin/env python
# *-* coding: UTF-8 *-*

import matplotlib.pyplot as plt

# 定义文本框 和 箭头格式 【 sawtooth 波浪方框, round4 矩形方框 , fc表示字体颜色的深浅 0.1~0.9 依次变浅，没错是变浅】
decision_node = dict(boxstyle = 'sawtooth', fc = '0.8')
leaf_node = dict(boxstyle = 'round4', fc = '0.8')
arrow_args = dict(arrowstyle = '<-')

# 
def plot_node(node_txt, center_pt, parent_pt, node_type):
    # 调用ax1对象的.annotate()方法添加注释
    create_plot.ax1.annotate(node_txt, xy = parent_pt, xycoords = 'axes fraction', 
                            xytext = center_pt, textcoords = 'axes fraction', va = 'center', ha = 'center', 
                            bbox = node_type, arrowprops = arrow_args)

# 绘图
def create_plot():
    #创建figure画图对象
    fig = plt.figure(1, facecolor = 'white')
    # figure画布清空
    fig.clf()
    # 创建子图，并赋值给ax1（函数名.变量名 可在函数外调用）
    create_plot.ax1 = plt.subplot(111, frameon = False)
    # 绘制决策节点
    plot_node('决策节点', (0.5, 0.1), (0.1, 0.5), decision_node)
    # 绘制叶节点
    plot_node('叶节点', (0.8, 0.1), (0.3, 0.8), leaf_node)
    
    plt.show()

# 返回传入子树的叶节点数
def get_num_leafs(my_tree):
    """"
    Des:
        返回传入子树的叶节点树
    Args：
        my_tree -- 传入的子树，为字典嵌套结构
    Return：
        num_leafs -- 叶节点数目
    思路：
        1. 字典嵌套结构，父节点为键，多个分支子节点为字典类型值
        2. 取出当前键的字典类型值
        3. 遍历字典中的所有键
        4. 判断所有键对应的值是否为字典
        5. 若是字典类型，则递归函数，将该字典作为实参传递入该函数
        6. 若否，则叶节点变量计数为1
    """
    num_leafs = 0
    # 取出字典嵌套树结构的第一个键（第一个决策节点）
    first_str = list(my_tree.keys())[0]
    # 取出第一个键对应的第一层嵌套字典（既第一个决策树）
    second_dict = my_tree[first_str]
    # 遍历树结构所有分支
    for key in second_dict.keys():
        # 判断依据键取出的子字典是否为字典类型，若是，则递归，将该子字典传递至本函数，返回其叶节点数
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs
        

def get_tree_depth(my_tree):
    """
    Des：
        返回传入子树的分支层数
    Args：
        my_tree -- 传入的字数，为字典嵌套结构
    Return：
        max_depth --> 最大分支深度
    思路：
        1. 遍历当前子树所有分支，判断分支是否为子树
        2. 若是，则递归，且计数器+1
        3. 若否，则计数器为1
        4. 返回深度最大值
    """
    max_depth = 0
    first_key = list(my_tree.keys())[0]
    first_value = my_tree[first_key]
    # 遍历当前子树的各分支
    for key in first_value.keys():
        depth = 0
        if type(first_value[key]).__name__ == 'dict':
            depth = 1 + get_tree_depth(first_value[key])
        else:
            depth = 1
        if max_depth < depth:
            max_depth = depth
    return max_depth
    
def plot_mid_text(center_pt, parent_pt, text_str):
    x_mid = (parent_pt[0] - center_pt[0]) / 2.0 + center_pt[0]
    y_mid = (parent_pt[1] - center_pt[1]) / 2.0 + center_pt[1]
    create_plot.ax1.text(x_mid, y_mid, text_str)
    
def plot_tree(my_tree, parent_pt, node_txt):
    num_leafs = get_num_leafs(my_tree)
    depth = get_tree_depth(my_tree)
    first_str = list(my_tree.keys())[0]
    center_pt = (plot_tree.xOff + (1.0 + float(num_leafs)) / 2.0 / plot_tree.totalW, plot_tree.yOff)
    
    plot_mid_text = (center_pt, parent_pt, node_txt)
    plot_node(first_str, center_pt, parent_pt, decision_node)
    
    second_dict = my_tree[first_str]
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD
    
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], center_pt, str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff), center_pt, leaf_node)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), center_pt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD
            
def create_plot(in_tree):
    fig = plt.figure(1, facecolor = 'white')
    fig.clf()
    axprops = dict(xticks = [], yticks = [])
    
    create_plot.ax1 = plt.subplot(111, frameon = False, **axprops)
    plot_tree.totalW = float(get_num_leafs(in_tree))
    plot_tree.totalD = float(get_tree_depth(in_tree))
    plot_tree.xOff = -0.5 / plot_tree.totalW
    plot_tree.yOff = 1.0
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()
    
