import numpy as np

class DecisionStump:
    def __init__(self):
        self.polarity = 1          # 极性
        self.feature_idx = None    # 特征索引
        self.threshold = None      # 阈值
        self.alpha = None          # 系数

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators  # 弱分类器数量
        self.stumps = []                  # 弱分类器列表
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # 初始化权重为 1/N
        w = np.full(n_samples, 1/n_samples)
        
        for i in range(self.n_estimators):
            # 寻找最佳决策树桩
            stump = DecisionStump()
            min_error = float('inf')
            for feature_idx in range(n_features):
                feature_values = np.expand_dims(X[:, feature_idx], axis=1)
                thresholds = np.unique(feature_values)
                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X[:, feature_idx] < threshold] = -1
                    error = np.sum(w[y != predictions])
                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    if error < min_error:
                        min_error = error
                        stump.polarity = p
                        stump.feature_idx = feature_idx
                        stump.threshold = threshold
            
            # 计算系数 alpha
            stump.alpha = 0.5 * np.log((1 - min_error) / (min_error + 1e-10))
            
            # 更新权重
            predictions = np.ones(n_samples)
            negative_idx = (stump.polarity * X[:, stump.feature_idx] < stump.polarity * stump.threshold)
            predictions[negative_idx] = -1
            w *= np.exp(-stump.alpha * y * predictions)
            w /= np.sum(w)
            
            # 保存决策树桩
            self.stumps.append(stump)
    
    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)
        for stump in self.stumps:
            predictions = np.ones(n_samples)
            negative_idx = (stump.polarity * X[:, stump.feature_idx] < stump.polarity * stump.threshold)
            predictions[negative_idx] = -1
            y_pred += stump.alpha * predictions
        y_pred = np.sign(y_pred)
        return y_pred.astype(int)

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    X = []
    y = []
    for d in data:
        d = d.strip()  # 去除换行符
        d = d[1:-1]  # 去除前后中括号
        d = d.split()  # 按空格分割字符串
        x = [float(v) for v in d[:-1]]
        X.append(x)
        y.append(float(d[-1]))
    y = np.array(y)
    y[y == 0] = -1
    return np.array(X), y


# 训练模型
def train():
    X_train, y_train = load_data('./data/train.txt')
    adaboost = AdaBoost(n_estimators=100)
    adaboost.fit(X_train, y_train)
    return adaboost

# 测试模型
def test(adaboost):
    X_test, y_test = load_data('./data/test.txt')
    print("y_test:", y_test)
    y_pred = adaboost.predict(X_test)
    print("y_pred:", y_pred)
    accuracy = np.mean(y_pred == y_test)
    return accuracy

if __name__ == '__main__':
    adaboost = train()
    accuracy = test(adaboost)
    print('Accuracy:', accuracy)