import numpy as np
from typing import List
from hw1_knn import KNN
import collections

# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    branches = np.array(branches)
    total_ele = np.sum(branches)
    # # entropy_child = np.sum([(branches[i]/total_ele[i]*np.log2(branches[i]/total_ele[i]) for i in range(len(total_ele)))])
    # entropy_child = [(branches[i] / total_ele[i]) * np.log2(branches[i] / total_ele[i] if branches[i]>0 else 0) for i in range(len(total_ele)) if total_ele[i]!=0]
    # entropy = np.sum([(entropy_child[i] * (total_ele[i]/np.sum(branches))) for i in range(len(entropy_child))])
    # info_gain = S+entropy
    # return info_gain
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
    queue = collections.deque([root])
    res = []
    while len(queue):
        pred_class = []
        for i in range(len(queue)):
            node = queue.popleft()
            pred_class.append(node)
            for child in node.children:
                queue.append(child)
            res.append(pred_class)
    return res


def traverse(node, search_node, is_split):
    if node == None:
        return node

    if node == search_node:
        node.splittable = is_split
        return node

    for child in node.children:
        traverse(child, search_node, is_split)


def accuracy_score(y_pred, y_test):
    # print(y_pred, y_test)
    print("***", len(y_pred), len(y_test))
    correct_count = sum([x==y for x, y in zip(y_pred, y_test)])
    score = correct_count/len(y_pred)
    print('score', score)
    return score


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # # decisionTree
    # # X_test: List[List[any]]
    # # y_test: List
    no_levels = tree_traversal(decisionTree.root_node)
    no_levels.reverse()
    # val_test = [[X_test[x][y] for y in range(len(X_test[0]))] for x in range(len(X_test))]
    init_accuracy = accuracy_score(decisionTree.predict(X_test), y_test)

    for level in no_levels:
        for node in level:
            # val_test = [[X_test[x][y] for y in range(len(X_test[0]))] for x in
            #             range(len(X_test))]
            # init_accuracy = compute_accuracy(decisionTree.predict(val_test),
            #                                  y_test)
            if node.splittable:
                traverse(decisionTree.root_node, node, False)
                # val_test = [[X_test[x][y] for y in range(len(X_test[0]))]
                #                for x in range(len(X_test))]
                y_pred = decisionTree.predict(X_test)
                accuracy = accuracy_score(y_pred, y_test)
                if accuracy <= init_accuracy:
                    traverse(decisionTree.root_node, node, True)
                else:
                    node.children = []
                    init_accuracy = accuracy


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
    # assert len(real_labels) == len(predicted_labels)
    # tp,fp,tn,fn = 0,0,0,0
    # print('inside f1')
    # for i in range(len(real_labels)):
    #     if predicted_labels[i] == 1 and real_labels[i] == 1:
    #         tp += 1
    #     elif predicted_labels[i] == 1 and real_labels[i] == 0:
    #         fp += 1
    #     elif predicted_labels[i] == 0 and real_labels[i] == 0:
    #         tn += 1
    #     elif predicted_labels[i] == 0 and real_labels[i] == 1:
    #         fn += 1
    # print('tp', tp,fp,fn)
    # if tp == fp == fn == 0:
    #     return 0
    # precision = tp/(tp + fp)
    # recall = tp/(tp + fn)
    # f1 = (2 * precision * recall)/(precision+recall)
    # print('precision,recal', precision, recall)
    # return f1
    # print('inside f1')
    assert len(real_labels) == len(predicted_labels)

    tp = sum([x == 1 and y == 1 for x, y in zip(real_labels, predicted_labels)])
    fp = sum([x == 0 and y == 1 for x, y in zip(real_labels, predicted_labels)])
    fn = sum([x == 1 and y == 0 for x, y in zip(real_labels, predicted_labels)])
    if 2 * tp + fp + fn == 0:
        return 0
    f1 = 2 * tp / float(2 * tp + fp + fn)
    return f1


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    dist = [(p - q) ** 2 for p, q in zip(point1, point2)]
    dist = np.sqrt(sum(dist))
    return dist


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    distance = [(x * y) for x, y in zip(point1, point2)]
    distance = sum(distance)
    return distance

def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    dist = [(p - q) ** 2 for p, q in zip(point1, point2)]
    dist = -0.5 * sum(dist)
    return -np.exp(dist)


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
        # print(name, distance_funcs[name], func)
        # best_f1_score, k_val = -1, 0
        for k in range(1,30,2):
            model = KNN(k=k, distance_function=func)
            model.train(Xtrain, ytrain)
            val_f1 = f1_score(yval, model.predict(Xval))
            if val_f1 > best_f1_score:
                best_f1_score, best_k = val_f1, k
                best_func = name
    #         print('best_func, best_k, best f1, val', name, best_k, k, best_f1_score, val_f1)
    #     print('**best_func, best_k', best_func, best_k)
    # print('****best_func, best_k', best_func, best_k)
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
            for k in range(1, 30, 2):
                model = KNN(k=k, distance_function=func)
                model.train(scale_train, ytrain)
                val_f1 = f1_score(yval, model.predict(scale_val))
                if val_f1 > best_f1_score:
                    best_f1_score, best_k = val_f1, k
                    best_func = name
                    best_scaler = scaler_name
                    best_scale_class = scale_class
    #             print('best_func, best_k, best f1, val', name, best_k, k,
    #                   best_f1_score, val_f1)
    #     print('**best_func, best_k, best_scale', best_func, best_k, best_scaler)
    # print('****best_func, best_k, best_scale', best_func, best_k, best_scaler)

    # best_scale_class = scaling_classes[best_scaler]
    scale_train = best_scale_class(Xtrain)
    best_model = KNN(k=best_k, distance_function=distance_funcs[best_func])
    # scale_val = scale_class(Xval)
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
        # normalized_vector = []
        # for feature in features:
        #     if all(item == 0 for item in feature):
        #         normalized_vector.append(feature)
        #     else:
        #         den = [x*x for x in feature]
        #         den = np.sqrt(den)
        #         vector = [val / den for val in feature]
        #         normalized_vector.append(vector)
        # return normalized_vector
        # norm_features = []
        # for x in features:
        #     sum = 0
        #     norm_feature_vec = []
        #     # if all the features in x are 0, then all_zeroes variable would store True else False
        #     all_zeroes = all(ft == 0 for ft in x)
        #     # below if condition avoids feature vector with zeroes from being normalized
        #     if (not all_zeroes):
        #         for i in range(len(x)):
        #             sum += x[i] * x[i]
        #         for i in range(len(x)):
        #             norm_feature_vec.append(x[i] / np.sqrt(sum))
        #     norm_features.append(norm_feature_vec)
        # return norm_features

        normalized = []
        for sample in features:
            if all(x == 0 for x in sample):
                normalized.append(sample)
            else:
                denom = float(np.sqrt(inner_product_distance(sample, sample)))
                sample_normalized = [x / denom for x in sample]
                normalized.append(sample_normalized)

        return normalized


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
