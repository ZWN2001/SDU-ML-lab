import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# 定义节点类
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # 分割特征
        self.threshold = threshold  # 分割阈值
        self.left = left  # 左子树
        self.right = right  # 右子树
        self.value = value  # 叶节点取值


# 定义决策树分类器
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth  # 决策树最大深度
        self.min_samples_split = min_samples_split  # 内部节点最小样本数
        self.min_samples_leaf = min_samples_leaf  # 叶节点最小样本数
        self.root = None  # 决策树的根节点

    # 计算基尼指数
    def gini(self, y):
        classes = np.unique(y)
        n_sample = y.shape[0]
        gini = 0
        for cls in classes:
            gini += (np.sum(y == cls) / n_sample) ** 2
        return 1 - gini

    # 计算信息熵
    def entropy(self, y):
        classes = np.unique(y)
        n_sample = y.shape[0]
        entropy = 0
        for cls in classes:
            ratio = np.sum(y == cls) / n_sample
            entropy -= ratio * np.log2(ratio + 1e-9)
        return entropy

    # 计算信息增益
    def information_gain(self, x, y, feature, threshold):
        mask = x[:, feature] < threshold
        y_left, y_right = y[mask], y[~mask]
        if len(y_left) == 0 or len(y_right) == 0:
            return 0
        info_gain = self.entropy(y) - (len(y_left) / len(y)) * self.entropy(y_left) - \
                    (len(y_right) / len(y)) * self.entropy(y_right)
        return info_gain

    # 按照最优特征和阈值分割数据集
    def split(self, x, y):
        max_info_gain = 0
        best_feature, best_threshold = None, None
        n_sample, n_feature = x.shape
        for feature in range(n_feature):
            thresholds = np.unique(x[:, feature])
            for threshold in thresholds:
                info_gain = self.information_gain(x, y, feature, threshold)
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_feature = feature
                    best_threshold = threshold
        if best_feature is None:
            return None, None, None
        mask = x[:, best_feature] < best_threshold
        X_left, y_left = x[mask], y[mask]
        X_right, y_right = x[~mask], y[~mask]
        return best_feature, best_threshold, X_left, y_left, X_right, y_right

    # 计算叶节点取值
    def leaf_value(self, y):
        classes, counts = np.unique(y, return_counts=True)
        idx = np.argmax(counts)
        return classes[idx]

    # 创建决策树
    def build_tree(self, X, y, depth=0):
        n_sample, n_feature = X.shape
        n_cls = len(np.unique(y))
        # 如果样本全部属于同一个类别或深度达到最大深度或样本数量太少则创建叶节点
        if n_cls == 1 or depth == self.max_depth or n_sample < self.min_samples_split:
            leaf_value = self.leaf_value(y)
            return Node(value=leaf_value)
        # 按照最优特征和阈值分割数据集
        feature, threshold, X_left, y_left, X_right, y_right = self.split(X, y)
        # 如果分割后的样本数量太小则创建叶节点
        if X_left is None:
            leaf_value = self.leaf_value(y)
            return Node(value=leaf_value)
        # 如果分割后的样本数量少于叶节点最小样本数则创建叶节点
        if len(X_left) < self.min_samples_leaf or len(X_right) < self.min_samples_leaf:
            leaf_value = self.leaf_value(y)
            return Node(value=leaf_value)
        # 创建分割节点
        node = Node(feature, threshold)
        # 递归创建左右子树
        node.left = self.build_tree(X_left, y_left, depth + 1)
        node.right = self.build_tree(X_right, y_right, depth + 1)
        return node

    # 使用训练数据集创建决策树
    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    # 预测单个样本的类别
    def _predict(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] < node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)

    # 对测试数据集进行预测
    def predict(self, X_test):
        y_pred = []
        for i in range(X_test.shape[0]):
            y_pred.append(self._predict(X_test[i], self.root))
        return np.array(y_pred)


if __name__ == '__main__':
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25)
    clf = DecisionTree(max_depth=3, min_samples_split=2, min_samples_leaf=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print("测试集准确率：", accuracy)
