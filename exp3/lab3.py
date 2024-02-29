import numpy as np
import math

# 加载数据集
def data_load():
    data = np.loadtxt("data//lenses_data.txt", dtype = int, usecols = [1, 2, 3, 4, 5])
    return data

# 计算x的数量
def cal_num(x, lable):
    num = 0
    for i in x:
        num += 1 if i == lable else 0
    return num

# 计算信息熵
def cal_infoentropy(x):
    infoentropy = 0
    for i in set(x):
        p_i = cal_num(x, i) / x.size
        infoentropy -= p_i * math.log(p_i, 2)
    return infoentropy

# 计算某特征的条件信息熵
def cal_conditional_infoentropy(feature, x):
    feature_lable = set(feature)
    lable = set(x)
    conditional_infoentropy = 0
    for i in feature_lable:
        p_i = cal_num(feature, i) / feature.size
        conditional_infoentropy += p_i *cal_infoentropy(x[feature == i])
    return conditional_infoentropy

# 计算某特征的信息增益率
def cal_infogainpercent(feature, x):
    # 信息增益
    infogain = cal_infoentropy(x) - cal_conditional_infoentropy(feature, x)
    # 信息增益率
    if cal_infoentropy(feature == 0):
        infogainpercent = 0
    else:
        infogainpercent = infogain / cal_infoentropy(feature)
    return infogainpercent

# 获取信息增益率最高的特征
def get_best_feature(data, lable, flag):
    index = 0
    best_infogainpercent = 0
    for row in range(0, data.shape[1]):
        infogainpercent = cal_infogainpercent(data[:,row], lable)
        if flag:
            print("第 %d 个特征的信息增益率为 %.17f" % (row + 1, infogainpercent),
                  file=out_file)
        if infogainpercent > best_infogainpercent:
            index = row
            best_infogainpercent = infogainpercent
    return index

# 对数据集分片
def data_split(data, index, value):
    split_dataset = []
    for col in data:
        if col[index] == value:
            split_coloum = col[:index].tolist()
            split_coloum.extend(col[index + 1:].tolist())
            split_dataset.append(split_coloum)
    split_dataset = np.array(split_dataset)
    return split_dataset

# 找到出现次数最多的标签
def most_lable(data):
    lable = list(set(data))
    most = ''
    mostcount = 0
    for item_lable in lable:
        count = 0
        for item_data in data:
            if item_lable == item_data:
                count += 1
        if count > mostcount:
            most = item_lable
        mostcount = count
    return most

# 创建决策树
def create_decision_tree(data, feature_label):
    labels = [item[-1] for item in data]

    # 若标签中都属于同一类，则直接返回
    if 1 == len(set(labels)):
        return labels[0]

    # 特征集为空或是特征集取值相同，则直接返回

    if len(data[0]) == 1 or len(set(data[:, 0])) == 1 or not feature_label:
        return most_label(labels)

    # 选择最优标签
    best_index = get_best_feature(data[:, 0:-1], np.array(labels), False)
    # 获取最优的标签
    best_label = feature_label[best_index]
    # 根据最优特征的标签生成树
    decision_tree = {best_label: {}}
    # 得到训练集中所有最优特征的标签
    feat_value = [item[best_index] for item in data]
    # 去掉重复值
    for value in set(feat_value):
        decision_tree[best_label][value] = create_decision_tree(
            data_split(data, best_index, value), split_feature(feature_label, best_index))
    return decision_tree

# 对特征分片
def split_feature(label, index):
    error_deal = []
    for i in range(0, len(label)):
        if i != index:
            error_deal.append(label[i])
    return error_deal

if __name__ == '__main__':
    feature_labels = ['年龄', '症状', '散光', '眼泪数量']
    dataset = data_load()
    out_file = open('output.txt', mode='a', encoding='utf-8')
    get_best_feature(dataset[:, 0:-1], dataset[:, -1], True)
    print(create_decision_tree(dataset, feature_labels), file=out_file)
    out_file.close()