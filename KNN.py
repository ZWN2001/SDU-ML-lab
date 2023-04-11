import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
        return self

    def predict(self, x_test):
        mlabels = []
        for x in x_test:
            distances = [np.sqrt(np.sum((x_train - x) ** 2)) for x_train in self.x_train]
            nearest = np.argsort(distances)[:self.k]
            top_k_y = [self.y_train[index] for index in nearest]
            d = {}
            for cls in top_k_y:
                d[cls] = d.get(cls, 0) + 1
            d_list = list(d.items())
            d_list.sort(key=lambda x: x[1], reverse=True)
            if len(d_list) > 0:
                mlabels.append(d_list[0][0])
        return mlabels


if __name__ == '__main__':
    iris = load_iris()

    max_k = 6
    min_k = 3
    times = 4
    result = np.zeros((max_k - min_k + 1, times), dtype=float)
    for i in range(0, times):
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25)
        for j in range(min_k, max_k + 1):
            clf = KNNClassifier(k=j)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = np.mean(y_pred == y_test)
            result[j - min_k][i] = accuracy

    # 折线图
    matplotlib.use('TkAgg')
    x = list(range(1, times + 1))  # 横坐标

    plt.plot(x, result[0], 'o-.', color='r', label="K=3")
    plt.plot(x, result[1], 'x-.', color='g', label="K=4")
    plt.plot(x, result[2], '*-.', color='b', label="K=5")
    plt.plot(x, result[3], '--', color='y', label="K=6")
    plt.xlabel("x")  # 横坐标名字
    plt.ylabel("accuracy")  # 纵坐标名字
    plt.legend(loc="best")  # 图例
    plt.show()
