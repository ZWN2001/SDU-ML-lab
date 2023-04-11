import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scipy.stats import norm


class NaiveBayes:

    def __init__(self):
        self.priors = None
        self.classes = None
        self.means = None
        self.stdevs = None

    def fit(self, X, Y):
        self.classes = np.unique(Y)
        self.means = np.zeros((len(self.classes), X.shape[1]))
        self.stdevs = np.zeros((len(self.classes), X.shape[1]))
        for i, cls in enumerate(self.classes):
            X_cls = X[Y == cls]
            self.means[i, :] = X_cls.mean(axis=0)
            self.stdevs[i, :] = X_cls.std(axis=0)
        self.priors = np.log(np.array([np.mean(Y == cls) for cls in self.classes]))

    def predict(self, X_test):
        mlabels = []
        for x in X_test:
            posteriors = []
            for i, cls in enumerate(self.classes):
                cond_prob = norm.logpdf(x, self.means[i, :], self.stdevs[i, :]).sum()
                posterior = self.priors[i] + cond_prob
                posteriors.append(posterior)
            mlabels.append(self.classes[np.argmax(posteriors)])
        return mlabels


if __name__ == '__main__':
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25)
    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print("测试集准确率：", accuracy)
