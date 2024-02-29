import numpy as np
import matplotlib.pyplot as plt

# 防止绘图出现中文乱码问题
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来显示负号
plt.rcParams['axes.unicode_minus'] = False

# 数据预处理
def data_process():
    file = "实验1_20090022058_朱甲文\data\housing_data.txt" # 数据文件路径
    data = np.fromfile(file, sep = " ") # 从指定的文件路径读入数据
    #print(data.shape) #data.shape = 7084(506*14)
    x = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    x_num = len(x)
    data = data.reshape(data.shape[0] // x_num, x_num)
    #print(data.shape) #data.shape = (506, 14)

    training_data = data[0 : 450] # 划分数据集，前450个样本为训练集
    
    maximums = training_data.max(axis=0)
    minimums = training_data.min(axis=0)
    avgs = training_data.sum(axis=0) / training_data.shape[0]

    # 对数据进行归一化
    # 如果不进行归一化处理，会产生梯度爆炸的问题，输出loss=nan
    for i in range(x_num):
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    training_data = data[0 : 450]
    test_data = data[450 : 506] # 测试集
    return training_data, test_data

class NetWork(object):
    # 初始化
    def __init__(self, num_weights):
        np.random.seed(0)
        self.w = np.random.randn(num_weights, 1)
        self.b = 0
    # 前向运算
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
    # 计算损失值(均方误差损失函数)
    def loss(self, z, y):
        error = z-y
        cost = error * error
        # 把所有样本的cost相加，求平均
        cost = np.mean(cost)
        return cost
    # 计算梯度
    def gradient(self, x, y, z):
        gradient_w = (z-y) * x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]

        gradient_b = z-y
        gradient_b = np.mean(gradient_b)
        return gradient_w, gradient_b
    # 更新参数
    def update(self, gradient_w, gradient_b, eta = 0.01):
        self.w -= eta * gradient_w
        self.b -= eta * gradient_b
    # 模型训练
    def train(self, training_data, num_epochs, batch_size=10, eta=0.01):
        losses = []
        n = len(training_data)
        for epoch_id in range(num_epochs):
            # 在每轮迭代开始之前，将训练数据的顺序随机打乱
            # 然后再按每次取batch_size条数据的方式取出
            np.random.shuffle(training_data)

            # 将训练数据拆分
            mini_batches = [training_data[k: k+ batch_size] for k in range(0, n, batch_size)]

            for iter_id, mini_batch in enumerate(mini_batches):
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]

                a = self.forward(x)
                L = self.loss(a, y)
                gradient_w, gradient_b = self.gradient(x, y, a)
                self.update(gradient_w, gradient_b, eta)
                losses.append(L)
                print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.
                      format(epoch_id, iter_id, L))

        return losses

if __name__ == '__main__':
    training_data, test_data = data_process()
    # x的形状是(404, 13)
    # y的形状是(404, 1)
    x = training_data[:, : -1]
    y = training_data[:, -1:]

    net = NetWork(13)
    num_epoches = 1000
    losses = net.train(training_data, num_epochs=num_epoches, batch_size=100, eta=0.01)
    
    # 训练结果可视化
    # 画出均方误差损失函数的变化趋势
    plot_x = np.arange(len(losses))
    plot_y = np.array(losses)
    plt.plot(plot_x, plot_y)
    plt.title("均方误差损失函数变化趋势")
    plt.show()

    # 画出预测值与真实值的对比
    tx = test_data[:, : -1]
    ty = test_data[:, -1:]
    y_predict=net.forward(tx)
    plot_x=np.arange(len(y_predict))
    plot_y=np.array(y_predict)
    plt.plot(plot_x,plot_y)
    plot_y1=np.array(ty)
    plt.plot(plot_x,plot_y1)
    plt.legend(['predict','true'],loc='upper left')
    plt.ylim([-0.7,0.5])
    plt.title("预测值与真实值对比")
    plt.show()

    # 画出预测误差
    plot_y2 = np.array(ty - y_predict)
    plt.plot(plot_x, plot_y2)
    plt.ylabel('predict error')
    plt.title("预测误差图")
    plt.show()