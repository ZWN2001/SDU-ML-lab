import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SVM:

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma=0.1, tol=1e-3, max_iter=100):
        self.C = C  # 惩罚参数
        self.kernel = kernel  # 核函数类型,这里只写了rbf
        self.degree = degree  # 多项式核的阶数
        self.gamma = gamma  # 核函数参数
        self.tol = tol  # 精度
        self.max_iter = max_iter  # 最大迭代次数，-1表示无限制
        self.alpha = None  # 拉格朗日乘子
        self.b = None  # 截距
        self.X = None  # 训练样本
        self.y = None  # 标签

    def fit(self, X1, y1):
        assert len(X1) == len(y1)
        self.X = X1
        self.y = y1
        self.b = 0
        n_samples, n_features = X1.shape
        # 计算核矩阵
        K = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                diff = X1[i] - X1[j]
                K[i, j] = np.exp(-self.gamma * np.dot(diff, diff))

        # 初始化拉格朗日乘子
        alpha = np.zeros(n_samples)
        # 外层循环
        for it in range(self.max_iter):
            alpha_old = np.copy(alpha)
            for i in range(n_samples):
                # 计算预测值和误差
                E_i = np.sum(alpha * self.y * K[:, i]) + self.b - self.y[i]
                # 计算是否满足KKT条件
                if (self.y[i] * E_i < -self.tol and alpha[i] < self.C) or (self.y[i] * E_i > self.tol and alpha[i] > 0):
                    # 随机选择另一个拉格朗日乘子
                    j = np.random.randint(n_samples)
                    while j == i:
                        j = np.random.randint(n_samples)
                    # 计算预测值和误差
                    E_j = np.sum(alpha * self.y * K[:, j]) + self.b - self.y[j]
                    # 保存alpha的旧值
                    alpha_i_old, alpha_j_old = alpha[i], alpha[j]

                    if self.y[i] == self.y[j]:
                        L = max(0, alpha[j] + alpha[i] - self.C)
                        H = min(self.C, alpha[j] + alpha[i])
                    else:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    if L == H:
                        continue
                    # 计算eta
                    eta = K[i, i] + K[j, j] - 2 * K[i, j]
                    if eta <= 0:
                        continue
                    # 计算未裁剪的新alpha
                    alpha[j] += self.y[j] * (E_i - E_j) / eta
                    alpha[j] = max(L, min(alpha[j], H))
                    # 检查alpha[j]是否有足够的变化量
                    if abs(alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    # 更新alpha[i]
                    alpha[i] += self.y[i] * self.y[j] * (alpha_j_old - alpha[j])
                    # 更新b
                    b1 = self.b - E_i - self.y[i] * (alpha[i] - alpha_i_old) * K[i, i] \
                         - self.y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - E_j - self.y[i] * (alpha[i] - alpha_i_old) * K[i, j] \
                         - self.y[j] * (alpha[j] - alpha_j_old) * K[j, j]
                    if 0 < alpha[i] < self.C:
                        self.b = b1
                    elif 0 < alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
            # 判断是否达到收敛条件
            diff = np.linalg.norm(alpha - alpha_old)
            if diff < self.tol:
                break
        self.alpha = alpha

    def predict(self, X1):
        assert self.alpha is not None and self.X is not None and self.y is not None
        n_samples, n_features = X1.shape
        y_pred_result = []

        for i in range(n_samples):
            s = 0
            for alpha, xi, yi in zip(self.alpha, self.X, self.y):
                s += alpha * yi * np.exp(-self.gamma * np.linalg.norm(xi - X1[i]) ** 2)
            y_pred_result.append(np.sign(s + self.b))

        return np.array(y_pred_result)


if __name__ == '__main__':
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    y_pred = np.zeros_like(y_test)
    clf = SVM(C=0.6, gamma=0.8, max_iter=500)
    for k in range(3):
        # 将k类样本设为+1，其他样本设为-1
        y_train_ovr = np.where(y_train == k, 1, -1)
        clf.fit(X_train, y_train_ovr)
        y_test_ovr = np.where(y_test == k, 1, -1)
        y_pred_ovr = clf.predict(X_test)
        # 将+1类预测结果转化为k
        y_pred_ovr[y_pred_ovr > 0] = k
        y_pred[y_test_ovr == 1] = y_pred_ovr[y_test_ovr == 1]

    acc = np.mean(y_pred == y_test)
    print('Accuracy:', acc)



