import numpy as np
from typing import List
from hw1_knn import KNN

# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    raise NotImplementedError


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    raise NotImplementedError


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
    # dist = [(p - q) ** 2 for p, q in zip(point1, point2)]
    # return sum(dist)
    sum = 0
    for i in range(len(point1)):
        sum += point1[i] * point2[i]
    return sum

def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    dist = [(p - q) ** 2 for p, q in zip(point1, point2)]
    dist = -0.5* sum(dist)
    return -np.exp(dist)


def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    num = sum(p*q for p,q in zip(point1, point2))
    den1 = np.sqrt(sum([p*p for p in point1]))
    den2 = np.sqrt(sum([q*q for q in point2]))
    return 1-(num/(den1*den2))


# TODO: select an instance of KNN with the best f1 score on validation dataset
def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    print('inside model_selection')
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
            print('best_func, best_k, best f1, val', name, best_k, k, best_f1_score, val_f1)
        print('**best_func, best_k', best_func, best_k)
    print('****best_func, best_k', best_func, best_k)
    return KNN(best_k, distance_funcs[best_func]), best_k, best_func


# TODO: select an instance of KNN with the best f1 score on validation dataset, with normalized data
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
            print(scaler_name, name, distance_funcs[name], func)
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
                print('best_func, best_k, best f1, val', name, best_k, k,
                      best_f1_score, val_f1)
        print('**best_func, best_k, best_scale', best_func, best_k, best_scaler)
    print('****best_func, best_k, best_scale', best_func, best_k, best_scaler)
    return KNN(best_k, distance_funcs[best_func]), best_k, best_func, best_scaler


class NormalizationScaler:
    def __init__(self):
        pass

    #TODO: normalize data
    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        normalized_vector = []
        for feature in features:
            if all(item == 0 for item in feature):
                normalized_vector.append(feature)
            else:
                den = [x*x for x in feature]
                den = np.sqrt(den)
                vector = [val / den for val in feature]
                normalized_vector.append(vector)
        return normalized_vector
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
        #             norm_feature_vec.append(x[i] / sum)
        #     norm_features.append(norm_feature_vec)
        # return norm_features


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
        self.count = 0
        self.min_max_values = []
        self.min, self.max = None, None

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        feat_array = np.array(features)
        if self.min is None or self.max is None:
            self.min = np.amin(feat_array, axis=0)
            self.max = np.amax(feat_array, axis=0)
        normalized = (feat_array - self.min) / (self.max - self.min)

        return normalized.tolist()






