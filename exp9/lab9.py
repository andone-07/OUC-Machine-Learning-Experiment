import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data = []
with open('./data/data.txt', 'r') as file:
    for line in file:
        line = line.strip()
        line = line[1:-1]
        x, y = line.split()
        data.append([float(x), float(y)])
data = np.array(data)

# 标准化数据
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
X = (data - mean) / std

# 计算协方差矩阵
cov_matrix = np.cov(X.T)

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 选择最大特征值对应的特征向量作为主成分
principal_component = eigenvectors[:, np.argmax(eigenvalues)]

# 将数据投影到主成分上
projected_data = np.dot(X, principal_component)

# 绘制原始数据和投影后的数据
plt.scatter(X[:, 0], X[:, 1], alpha=0.8, label='Original Data')
plt.plot(projected_data, np.zeros_like(projected_data), 'r', label='Projected Data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()