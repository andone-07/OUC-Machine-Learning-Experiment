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

# 定义K-means函数
def k_means(data, k, max_iters=100):
    # 随机初始化聚类中心
    centers = data[np.random.choice(range(len(data)), k, replace=False)]

    for _ in range(max_iters):
        # 计算每个样本到聚类中心的距离
        distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)

        # 分配样本到最近的聚类中心
        labels = np.argmin(distances, axis=1)

        # 更新聚类中心
        new_centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # 判断是否收敛
        if np.all(centers == new_centers):
            break

        centers = new_centers

    return labels, centers

# 聚类数量
k = 3

# 运行K-means算法
labels, centers = k_means(data, k)

# 绘制聚类结果
colors = ['r', 'g', 'b']
for i in range(k):
    cluster_data = data[labels == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[i], label=f'Cluster {i+1}')

plt.scatter(centers[:, 0], centers[:, 1], c='k', marker='x', label='Centroids')
plt.xlabel('x')
plt.ylabel('y')
plt.title('K-means Clustering')
plt.legend()
plt.show()
