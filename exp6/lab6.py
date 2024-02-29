import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

Label_num = 10

train_dir = './data/traindata'
train_num = [189,198,195,199,186,187,195,201,180,204]

test_dir  = './data/testdata'
test_num = [87,97,92,85,114,108,87,96,91,89]

P = 0.5
# 超参数
m = 3

Train_data = []
Train_label = []
Test_data = []
Test_label = []

"""
读取训练数据和测试数据并对其进行预处理
Args:
    filename:数据路径
Returns:
    data: 经过处理的测试数据，一个列表，其中每个元素为一个长度为784的一维列表，表示一张图片的像素值
    label: 经过处理的测试标签，一个列表，其中每个元素为一个标签，表示一张图片所代表的数字
"""
# 读入训练集
def load_Train_Data(dir):
    # cnt = 0
    train_data = []
    train_label = []
    # 按照数字读入
    for i in range(0,10):
        # 按照数字的样本读入
        for j in range(0,train_num[i]):
            train_label.append(i)
            file_dir = dir+ '\\'+ str(i) +'_' + str(j) +'.txt'
            data = np.loadtxt(file_dir,dtype=str)
            # print(data)
            # cnt = cnt + 1
            tmp = []
            for m in range(0,32):
                # print(data[m])
                for n in data[m]:
                    num = int(n)
                    tmp.append(num)

            train_data.append(tmp)
    # print(cnt)
    return train_data,train_label

# 读入测试集
def load_Test_Data(dir):
    # cnt = 0
    test_data = []
    test_label = []
    for i in range(0, 10):
        for j in range(0, test_num[i]):
            test_label.append(i)
            file_dir = dir + '\\' + str(i) + '_' + str(j) + '.txt'
            data = np.loadtxt(file_dir, dtype=str)
            # print(data)
            # cnt = cnt + 1
            tmp = []
            for m in range(0, 32):
                # print(data[m])
                for n in data[m]:
                    num = int(n)
                    tmp.append(num)

            test_data.append(tmp)
    # print(cnt)
    return test_data, test_label

def cal_Likelihood(train_data,train_label):
    """
    计算训练数据的类条件概率和先验概率
    Returns:
        label_prob: 各个标签出现的先验概率，一个长度为Label_num的一维numpy数组
        feature_prob: 各个标签的每个像素的条件概率，一个形状为(Label_num, 784)的二维numpy数组
    """
    train_num = train_data.shape[0]
    train_dim = train_data.shape[1]
    label_prob =  np.zeros((Label_num,))
    feature = np.zeros((Label_num,train_dim))
    #print(label_prob.shape)
    label_stat = Counter(train_label)
    #print(label_stat)

    # 计算每个标签的先验概率
    for i in range(Label_num):
        label_prob[i] = label_stat[i] / train_num

    # 计算每个标签的每个像素的条件概率
    for i in range(Label_num):
        indexes = np.where(train_label == i)
        num_of_i = indexes[0].shape[0]
        for j in range(0,train_dim):
            times = 0
            #print(j)
            for index in indexes[0]:
                #print(train_data[index][j])
                if(train_data[index][j] == 0): #反正一个像素点的位置不过取值0或1，随便取一个进行计算即可
                    times += 1
                    #print(times)

            # 计算条件概率，采用拉普拉斯平滑避免概率为0的情况
            feature[i][j] = (times  + m*P ) / (num_of_i  + m)
    return label_prob, feature

# 预测测试数据的标签
def predict(test_data,label_prob,feature_prob):
    # 初始化每个标签的概率为1
    Possi = np.ones((Label_num,))
    # print(Possi)
    # 先乘上先验概率
    Possi = Possi * label_prob
    # print(Possi)
    for i in range(Label_num):
        feature_id = 0
        for data in test_data:
            '''
            被注释掉的是普通连乘
            没被注释的是对数似然
            二者结果一样，因为此处连乘没有出现下溢
            '''
            if(data == 0):
                #Possi[i] = Possi[i] * feature_prob[i][feature_id]
                Possi[i] = Possi[i] + np.log(feature_prob[i][feature_id])
            else:
                #Possi[i] = Possi[i] * (1 - feature_prob[i][feature_id])
                Possi[i] = Possi[i] + np.log((1 - feature_prob[i][feature_id]))
            feature_id += 1

    # 返回概率最大的标签
    predict_label = np.where(Possi == max(Possi))
    #print(Possi)
    return predict_label[0][0]

# 加载训练集和测试集
Train_data,Train_label = load_Train_Data(train_dir)
Train_data = np.asarray(Train_data)
Train_label = np.asarray(Train_label)

'''
label_prob: 每一个标签出现的概率，即先验概率
feature_prob：各个标签的每个像素的条件概率
             每一行的各列乘起来就是这一个标签的极大似然估计
'''

label_prob,feature_prob = cal_Likelihood(Train_data,Train_label)
print("各个标签出现的先验概率:\n{}".format(label_prob))
print("各个标签的每个像素的条件概率:\n{}".format(feature_prob))
Test_data,Test_label = load_Test_Data(test_dir)
Test_data = np.asarray(Test_data)
Test_label = np.asarray(Test_label)

acc = 0

for i in range(Test_data.shape[0]):
    final_label = predict(Test_data[i], label_prob, feature_prob)
    if(final_label == Test_label[i]):
       acc = acc + 1

# 精度
accuracy = (acc)/(Test_data.shape[0])
print("Accuracy rate: ",accuracy)