import numpy as np
import utils as Util
import math


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        print('in train')
        assert (len(features) > 0)
        self.feature_dim = len(features[0])
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
            # print ("feature: ", feature)
            # print ("pred: ", pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split


    #TODO: try to split current node
    def split(self):
        def get_entropy(labels):
            n = len(labels)
            unique_labels = np.unique(labels)
            sums = 0
            for label in unique_labels:
                no_instances = labels.count(label)
                p = float(no_instances) / float(n)
                sums += -(p * math.log(p, 2))
            return sums
        S = get_entropy(self.labels)
        for idx in range(len(self.features[0])):
            if not "info_gain" in locals():
                info_gain = float('-inf')
            values_at_dimensions = np.array(self.features)[:, idx]
            if None in values_at_dimensions:
                continue
            value_at_branches = np.unique(values_at_dimensions)
            branches = np.zeros((self.num_cls, len(value_at_branches)))
            for branch_ind in range(0, len(value_at_branches)):
                pred_vals = np.array(self.labels)[
                    np.where(values_at_dimensions == value_at_branches[branch_ind])]
                for eachPred in pred_vals:
                    branches[eachPred, branch_ind] += 1
            branches = branches.T
            # print(branches)
            IG = Util.Information_Gain(S,branches)
            if IG > info_gain:
                info_gain = IG
                self.dim_split = idx
                self.feature_uniq_split = value_at_branches.tolist()
        print('splitting at index', self.dim_split, value_at_branches)
        xi = np.array(self.features)[:, self.dim_split]
        x = np.array(self.features, dtype=object)
        x[:, self.dim_split] = None
        # x = np.delete(self.features, self.dim_split, axis=1)
        for val in self.feature_uniq_split:
            indexes = np.where(xi == val)
            x_new = x[indexes].tolist()
            y_new = np.array(self.labels)[indexes].tolist()
            child = TreeNode(x_new, y_new, self.num_cls)
            if np.array(x_new).size == 0 or all(
                            v is None for v in x_new[0]):
                child.splittable = False
            self.children.append(child)

        # split the child nodes
        for child in self.children:
            if child.splittable:
                child.split()

        return


    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        if self.splittable:
            # print(feature)
            idx_child = self.feature_uniq_split.index(feature[self.dim_split])
            return self.children[idx_child].predict(feature)
        else:
            return self.cls_max