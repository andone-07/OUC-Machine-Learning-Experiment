import numpy as np

# 加载数据集
def data_load(data_file):
    data = np.loadtxt(data_file, dtype = int, usecols = [1, 2, 3, 4, 5])
    return data

# 定义一个简单的K最近邻分类器
class KNNClassifier:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        y_pred = []
        for x in X:
            # 计算与训练集中样本的距离
            distances = [(self._euclidean_distance(x, xt), yt) for xt, yt in zip(self.X_train, self.y_train)]
            # 根据距离排序并选择最近的 k 个样本
            k_nearest = sorted(distances)[:self.k]
            # 统计 k 个样本中最常见的类别作为预测结果
            counts = {}
            for _, label in k_nearest:
                counts[label] = counts.get(label, 0) + 1
            predicted_label = max(counts, key=counts.get)
            y_pred.append(predicted_label)
        return y_pred
    
    def _euclidean_distance(self, x1, x2):
        return sum((a - b) ** 2 for a, b in zip(x1, x2)) ** 0.5

# 读取数据集文件
data_file = './data/lenses_data.txt'
data = data_load(data_file)
feature_labels = ['年龄', '症状', '散光', '眼泪数量']

# print(data)

# 将数据集拆分为训练集和测试集
train_size = 20
X_train = data[:train_size, :-1]
y_train = data[:train_size, -1]
X_test = data[train_size:, :-1]
y_test = data[train_size:, -1]

# 前向搜索算法选择最优特征集合
best_features = []
best_accuracy = 0.0

# 迭代所有特征
for i in range(X_train.shape[1]):
    # 添加当前特征到特征集合
    current_features = best_features + [i]
    
    # 使用当前特征集合训练模型
    model = KNNClassifier(k=3)
    model.fit(X_train[:, current_features], y_train)
    
    # 在测试集上进行预测
    y_pred = model.predict(X_test[:, current_features])
    
    # 计算准确率
    accuracy = sum(1 for y_true, y_pred in zip(y_test, y_pred) if y_true == y_pred) / len(y_test)
    
    # 如果当前特征集合的准确率更高，则更新最优特征集合和准确率
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_features = current_features

# 输出最优特征集合和测试集准确率
print("最优特征集合：", [feature_labels[i] for i in best_features])
print("测试集准确率：", best_accuracy)