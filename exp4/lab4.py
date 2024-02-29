import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 防止绘图出现中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据集
def load_data(file_name):
    data = np.loadtxt(file_name)
    data = pd.DataFrame(data,columns=["0","1","2"])
    #data.loc[data["2"].isin([0]),"2"] = -1
    data = np.asarray(data)
    return data

class layer(object):
    # 初始化
    def __init__(self,input_size,output_size,lr):
        self.input_size = input_size
        self.output_size =output_size
        self.lr = lr

    # 初始化权重矩阵和偏置矩阵
    def init_params(self):
        self.w_ = np.random.normal(loc=0, scale=0.01, size=(self.input_size, self.output_size))
        self.bias = np.zeros([1, self.output_size])
        #print(self.w_.shape)

    # 实现前向传播
    def forward(self,data):
        self.input = data
        # 全连接层的前向传播，计算输出结果
        self.output = self.input.dot(self.w_) + self.bias
        #print("params:",self.w_,self.bias)
        return self.output

    # 实现反向传播
    def backward(self,top_diff):
        # print(self.input.shape)
        # print(top_diff)
        self.input = self.input.reshape(1, -1)
        self.d_w = np.matmul(self.input.T, top_diff)
        #print(self.d_w)
        self.d_bias = np.matmul(np.ones([1, top_diff.shape[0]]), top_diff)
        bottom_diff = np.matmul(top_diff, self.w_.T)
        return bottom_diff

    # 更新权重矩阵和偏置矩阵
    def update(self,lr):
        self.w_ = self.w_ - lr * self.d_w
        self.bias = self.bias - lr * self.d_bias

    # 获取权重矩阵
    def getweight(self):
        return self.w_

    # 获取偏置矩阵
    def getbias(self):
        return self.bias

# 实现ReLU激活函数
class ReLu(object):
    def forward(self,input):
        self.input = input
        output = np.maximum(0, self.input)
        return output

    def backward(self,top_diff):
        return top_diff * (self.input>=0.)

# 实现激活函数Sigmoid
class Sigmoid(object):
    def forward(self,input):
        self.inputs = input
        output = 1/(1 + np.exp(-self.inputs))
        return output

    def backward(self,top_diff):
        index = self.inputs[0][0]
        f = 1 / (1 + np.exp(-index))
        return top_diff * f * (1 - f)

# 定义损失函数
class Loss(object):
    def forward(self,inputs,label):
        self.input = inputs
        self.label = label
        loss = ((inputs - label) ** 2) / 2
        return loss

    def backward(self):
        #print(self.input)
        #print(self.label)
        bottom_diff = self.input - self.label
        return bottom_diff

dir ='./data/perceptron_data.txt'

def sigmoid(input):
    output = 1 / (1 + np.exp(-input))
    return output

train_data = load_data(dir)

epoch = 1000
lr = 0.01

# 定义神经网络层
layer1 = layer(input_size=2,output_size=1,lr=lr)
layer1.init_params()
# relu1 = ReLu()
# layer2 = layer(input_size=2,output_size=2,lr=lr)
# layer2.init_params()
# relu2 = ReLu()
# layer3 = layer(input_size=2,output_size=1,lr=lr)
# layer3.init_params()
sigmoid1 = Sigmoid()
lossfun = Loss()

# 训练神经网络
for i in range(epoch):
    for vec in train_data:
        data = vec[:-1]
        label = vec[-1]
        #前向传播
        h1 =layer1.forward(data=data)
        prob = sigmoid1.forward(h1)
        #print("h3_2", h3)
        #计算损失
        loss = lossfun.forward(prob,label)

        #反向传播进行更新
        dloss = lossfun.backward()
        dh2 = sigmoid1.backward(dloss)
        #print("dh3_1",dh3)

        dh1 = layer1.backward(dh2)
        #print("dh1_2", dh1)
        layer1.update(lr)

    print('Epoch %d,loss: %.6f' % (i+1, loss),"param:w:{};b:{}".format(layer1.getweight(),layer1.getbias()))

# 得到训练得到的权重和偏置
w= layer1.getweight()
w = w.reshape(1,2)
print("w:", w)
b = layer1.getbias()
print("b:", b)
y = train_data[:, -1]
C1 = train_data[y == 0, :-1]
C2 = train_data[y == 1, :-1]

# 画出数据点和训练得到的超平面
plt.figure(figsize=(10,10))
x_values = np.arange(-5, 5)
y_values = -(w[0][0] * x_values + b)/w[0][1]
y_values = np.squeeze(y_values)
print("x_values:", x_values)
print("y_values:", y_values)

plt.plot(x_values, y_values, label='超平面')
plt.scatter(C1[:, 0],C1[:, 1], s=30, color='b', label='class 1')
plt.scatter(C2[:, 0],C2[:, 1], s=30, color='r', label='class 0')
plt.legend()
plt.show()