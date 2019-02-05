import numpy as np
from typing import List
from hw1_knn import KNN

# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    branches = np.array(branches)
    total_ele = np.sum(branches)
    total_in_branch = np.sum(branches, axis=1)
    prob_branch = total_in_branch / total_ele
    entropy = (branches.T / total_in_branch).T
    entropy = np.array([[-val * np.log2(val) if val > 0 else 0 for val in branch] for branch in entropy])
    entropy = np.sum(entropy, axis=1)
    entropy = np.sum(entropy * prob_branch)
    info_gain = S - entropy
    return info_gain


def tree_traversal(root):
    if not root:
        return []
    queue = []
    queue.append(root)
    res = []
    while len(queue):
        pred_level = []
        for i in range(len(queue)):
            node = queue.pop()
            pred_level.append(node)
            for child in node.children:
                queue.append(child)
            res.append(pred_level)
    return res


def traverse(node, search_node, is_split):
    if node == None:
        return node

    if node == search_node:
        node.splittable = is_split
        return node

    for child in node.children:
        traverse(child, search_node, is_split)


def compute_accuracy(y_pred, y_test):
    correct_count = sum([x==y for x, y in zip(y_pred, y_test)])
    score = correct_count/len(y_pred)
    return score


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # # decisionTree
    # # X_test: List[List[any]]
    # # y_test: List
    no_levels = tree_traversal(decisionTree.root_node)
    no_levels.reverse()
    init_accuracy = compute_accuracy(decisionTree.predict(X_test), y_test)

    for level in no_levels:
        for node in level:
            if node.splittable:
                traverse(decisionTree.root_node, node, False)
                y_pred = decisionTree.predict(X_test)
                accuracy = compute_accuracy(y_pred, y_test)
                if accuracy > init_accuracy:
                    node.children = []
                    init_accuracy = accuracy
                else:
                    traverse(decisionTree.root_node, node, True)


# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    assert len(real_labels) == len(predicted_labels)

    tp = sum([x == 1 and y == 1 for x, y in zip(real_labels, predicted_labels)])
    fp = sum([x == 0 and y == 1 for x, y in zip(real_labels, predicted_labels)])
    fn = sum([x == 1 and y == 0 for x, y in zip(real_labels, predicted_labels)])
    if 2 * tp + fp + fn == 0:
        return 0
    f1 = 2 * tp / float(2 * tp + fp + fn)
    return f1


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    distance = [(p - q) ** 2 for p, q in zip(point1, point2)]
    distance = np.sqrt(sum(distance))
    return distance


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    distance = [(x * y) for x, y in zip(point1, point2)]
    distance = sum(distance)
    return distance

def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    distance = [(p - q) ** 2 for p, q in zip(point1, point2)]
    distance = -0.5 * sum(distance)
    return -np.exp(distance)


def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    num = sum(p*q for p,q in zip(point1, point2))
    den1 = np.sqrt(sum([p*p for p in point1]))
    den2 = np.sqrt(sum([q*q for q in point2]))
    return 1-(num/(den1*den2))


def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    best_k, best_func, best_f1_score = 0, "", -1
    for name, func in distance_funcs.items():
        for k in range(1,30,2):
            model = KNN(k=k, distance_function=func)
            model.train(Xtrain, ytrain)
            val_f1 = f1_score(yval, model.predict(Xval))
            if val_f1 > best_f1_score:
                best_f1_score, best_k = val_f1, k
                best_func = name
    best_model = KNN(k=best_k, distance_function=distance_funcs[best_func])
    best_model.train(Xtrain, ytrain)
    return best_model, best_k, best_func


def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # scaling_classes: diction of scalers
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    # return best_scaler: best function choosed for best_model
    best_k, best_func, best_scaler, best_f1_score = 0, "", "", -1
    for scaler_name, scaler in scaling_classes.items():
        for name, func in distance_funcs.items():
            # print(scaler_name, name, distance_funcs[name], func)
            # best_f1_score, k_val = -1, 0
            scale_class = scaler()
            scale_train = scale_class(Xtrain)
            scale_val = scale_class(Xval)
            for k in range(1, 32, 2):
                model = KNN(k=k, distance_function=func)
                model.train(scale_train, ytrain)
                val_f1 = f1_score(yval, model.predict(scale_val))
                if val_f1 > best_f1_score:
                    best_f1_score, best_k = val_f1, k
                    best_func = name
                    best_scaler = scaler_name
                    best_scale_class = scale_class
    scale_train = best_scale_class(Xtrain)
    best_model = KNN(k=best_k, distance_function=distance_funcs[best_func])
    best_model.train(scale_train, ytrain)
    return best_model, best_k, best_func, best_scaler


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        normalized_array = []
        for sample in features:
            if all(x == 0 for x in sample):
                normalized_array.append(sample)
            else:
                denom = float(np.sqrt(inner_product_distance(sample, sample)))
                sample_normalized = [x / denom for x in sample]
                normalized_array.append(sample_normalized)

        return normalized_array


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """
    def __init__(self):
        self.min = None
        self.max = None

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        feature_array = np.array(features)
        if self.min is None or self.max is None:
            self.min = np.amin(feature_array, axis=0)
            self.max = np.amax(feature_array, axis=0)
        normalized_array = (feature_array - self.min) / (self.max - self.min)
        return normalized_array.tolist()
