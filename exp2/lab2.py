import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# 防止绘图出现中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据预处理
def data_process():
    file = "data//blood_data.txt" # 数据文件路径
    data = np.loadtxt(file, dtype=np.float32, delimiter=',') # 从指定的文件路径读入数据
    #print(data.shape)
    x = ['R', 'F', 'M', 'T', 'D']
    x_num = len(x)
    data = data.reshape([data.shape[0], x_num])
    #print(data.shape)

    training_data = data[0: 600]
    training_x, training_y = training_data[:, :(training_data.shape[1] - 1)], training_data[:, -1]
    testing_data = data[600: 748]
    testing_x, testing_y = testing_data[:, :(testing_data.shape[1] - 1)], testing_data[:, -1]
    # print(training_x, training_y, testing_x, testing_y)
    return training_x, training_y, testing_x, testing_y

class LDAModel(object):
    """
    线性判别分析
    """
    def __init__(self, data, target, d) -> None:
        self.data = data
        self.target = target
        self.d = d
        self.labels = set(target)
        # 列的平均值 μ
        # 所有样本的均值向量
        self.meanv = self.data.mean(0)
        self.new_data = None # 降维后的样本
        self.Swt_Sb = None
        self.w = None # 投影矩阵
        self.S_w = None # 类内散度矩阵
        self.S_b = None #类间散度矩阵
        self.S_t = None # 全局散度矩阵
        self.classify, self.class_meanv = None, None #第i类样本及其均值向量
    
    # 计算均值向量
    def comp_mean_vectors(self):
        self.classify, self.class_meanv = {}, {}
        for lable in self.labels:
            self.classify[lable] = self.data[self.target == lable]
            self.class_meanv[lable] = self.classify[lable].mean(0)

    # 计算全局散度矩阵
    def scatter_overall(self):
        self.S_t = np.dot((self.data - self.meanv).T, (self.data - self.meanv))

    # 计算类间散度矩阵
    def scatter_between(self):
        n_features = self.data.shape[1]
        print("n_features",n_features)
        self.S_b = np.zeros((n_features, n_features))

        for i in self.labels:
            class_i = self.classify[i] # 类别i样例的集合
            meanv_i = self.class_meanv[i] # 类别i的均值向量
            self.S_b += len(class_i) * np.dot((meanv_i - self.meanv).reshape(-1, 1),
                                              (meanv_i - self.meanv).reshape(1, -1))
        print("S_B:", self.S_b)

    # 计算类内散度矩阵
    def scatter_within(self):
        self.scatter_overall()
        self.scatter_between()
        self.S_w = self.S_t - self.S_b

    # 计算投影矩阵w
    def get_components(self):
        self.comp_mean_vectors()
        self.scatter_within()
        self.Swt_Sb = np.linalg.inv(self.S_w).dot(self.S_b)
        eig_vals, eig_vecs = np.linalg.eig(self.Swt_Sb)
        top_d = np.argsort(eig_vals[::-1])[:self.d]
        self.w = eig_vecs[:, top_d]
        print('EigVals: %s\n\nEigVecs: %s' % (eig_vals, eig_vecs))
        print('\nW: %s' % self.w)

    # 对样本进行降维
    def get_new_sample(self):
        self.get_components()
        self.new_data = np.dot(self.data, self.w)
    
    # 计算
    def get_result(self):
        self.get_new_sample()

def test():
    # 加载数据集
    training_x, training_y, testing_x, testing_y = data_process()
    # 创建模型
    model = LDAModel(training_x, training_y, 1)
    # 获取投影矩阵
    model.get_result()
    # 对训练集进行降维
    training_x_handled = model.new_data
    # 获取均值和方差
    gauss_dist = {}
    for i in model.labels:
        category = training_x_handled[training_y == i]
        loc = category.mean()
        scale = category.std()
        gauss_dist[i] = {'loc':loc, 'scale':scale}
    # 测试集降维
    testing_x_handled = np.dot(testing_x, model.w)
    pred_y = np.array([judge(gauss_dist, x) for x in testing_x_handled])
    
    # 绘图
    plt.scatter(training_x_handled[training_y == 0], np.ones((1, training_x_handled[training_y == 0].shape[0])),
                marker='o', color='red', label='not donate blood: 0', alpha=0.5)
    plt.scatter(training_x_handled[training_y == 1], np.zeros((1, training_x_handled[training_y == 1].shape[0])),
                marker='+', color='blue', label='donate blood: 1', alpha=0.5)
    plt.legend(loc="center right")
    plt.title("Training data(dimension reduction)") # 训练集降维的情况
    plt.ylabel("donate the blood(1) or not(0)")
    plt.xlabel("parameter after dimension reduction") # 降维后的参数
    plt.show()

    x = range(0, 148)
    plt.bar(x, testing_y, width = 0.3, color = '#ff4eff', label = 'test', alpha = 0.5)
    plt.bar(x, -pred_y, width = 0.3, color = '#ba2c01', label = 'pred', alpha = 0.5)
    plt.legend()
    # 真实值与预测值比较
    plt.title("Compare the true value with the predicted value\n \
    （accuracy：" + str(accuracy_score(testing_y, pred_y)) + ")")
    plt.ylabel("donate the blood(1) or not(0)")
    # 是否献血
    plt.xlabel("seq")
    # 序号
    plt.show()
    mylog = open('output.txt', mode='a+', encoding='utf-8')
    print("result(w)：", model.w.T, file=mylog)
    print("accuracy：", accuracy_score(testing_y, pred_y), file=mylog)
    mylog.close()
    # return accuracy_score(test_y, pred_y)

# 判断样本x属于哪个类别
def judge(gauss_dist, x):
    result = [[k, norm.pdf(x, loc = v['loc'], scale = v['scale'])] for k, v in gauss_dist.items()]
    result.sort(key = lambda s:s[1], reverse = True)
    return result[0][0]

# 计算精确度
def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis = 0) / len(y_true)
    return accuracy

if __name__ == "__main__":
    test()
    print("finish!")