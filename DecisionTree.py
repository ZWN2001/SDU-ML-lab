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

        # 剪枝函数

    def prune(self, node, X_val, y_val):
        if node.left is None and node.right is None:
            return False
        # 遍历左右子树，并记录下所有叶节点
        if node.left and not self.prune(node.left, X_val, y_val):
            return False
        if node.right and not self.prune(node.right, X_val, y_val):
            return False
        # 保存子树，接下来我们将尝试裁剪它
        tmp_left = node.left
        tmp_right = node.right
        # 裁剪子树，将其变为一个叶节点
        node.left = None
        node.right = None
        # 计算裁剪前的准确率
        y_pred = self.predict(X_val)
        accuracy_before = np.mean(y_pred == y_val)
        # 计算裁剪后的准确率
        leaf_value = self.leaf_value(y_val)
        y_pred = [leaf_value] * len(y_val)
        accuracy_after = np.mean(y_pred == y_val)
        if accuracy_before <= accuracy_after:
            # 裁剪后的准确率更高，则保留剪枝后的叶节点
            node.left = tmp_left
            node.right = tmp_right
            return False
        else:
            # 裁剪后的准确率更低，则保留未剪枝的节点
            return True

    # 使用训练数据集创建决策树
    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)  # 随机选择20%的数据作为验证集
        self.root = self.build_tree(X_train, y_train)
        self.prune(self.root, X_val, y_val)  # 剪枝

    # 预测单个样本的类别
    def _predict(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] < node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)

    # 剪枝后的预测函数
    def predict(self, X):
        assert self.root is not None, "build_tree at first"
        y = []
        for x in X:
            node = self.root
            while node.left:
                if x[node.feature] < node.threshold:
                    node = node.left
                else:
                    node = node.right
            y.append(node.value)
        return np.array(y)


if __name__ == '__main__':
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25)
    clf = DecisionTree(max_depth=3, min_samples_split=2, min_samples_leaf=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print("测试集准确率：", accuracy)
