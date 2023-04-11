import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
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

    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

    k_values = range(1, 31)
    kf = KFold(n_splits=5)

    mean_accuracy_values = []
    for k in k_values:
        accuracy_values = []
        for train_index, val_index in kf.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            knn = KNNClassifier(k=k)
            knn.fit(X_train_fold, y_train_fold)
            y_pred_val = knn.predict(X_val_fold)

            accuracy_values.append(accuracy_score(y_val_fold, y_pred_val))

        mean_accuracy = np.mean(accuracy_values)
        mean_accuracy_values.append(mean_accuracy)

    matplotlib.use('TkAgg')
    plt.plot(k_values, mean_accuracy_values)
    plt.title("Cross-Validation Accuracy over k")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.show()

    best_k = k_values[np.argmax(mean_accuracy_values)]
    print(f"Best value of k: {best_k}")

    knn = KNNClassifier(k=best_k)
    knn.fit(X_train, y_train)
    y_pred_test = knn.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Test accuracy with k={best_k}: {test_accuracy:.2f}")
