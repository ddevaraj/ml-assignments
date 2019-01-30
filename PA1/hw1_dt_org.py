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
        print('in split')
        def parent_entropy(labels):
            n = len(labels)
            unique_labels = np.unique(labels)
            sums = 0
            for label in unique_labels:
                no_instances = labels.count(label)
                p = float(no_instances) / float(n)
                sums += -(p * math.log(p, 2))
            return sums
        S = parent_entropy(self.labels)
        print('features, label, entropy', self.features, self.labels, S)
        for idx in range(len(self.features[0])):
            print('iteration: ', idx)
            if "info_gain" not in locals():
                info_gain = float('-inf')
            values_at_dimensions = np.array(self.features)[:, idx]
            print('vales at attribute', values_at_dimensions)
            if None not in values_at_dimensions:
                value_at_branches = np.unique(values_at_dimensions)
                branches = np.zeros((self.num_cls, len(value_at_branches)))
                print('value at branches', value_at_branches)
                for i, val in enumerate(value_at_branches):
                    y = np.array(self.labels)[np.where(values_at_dimensions == val)]
                    for yi in y:
                        branches[yi, i] += 1
                branches = branches.T
                # print(branches)
                IG = Util.Information_Gain(S,branches)
                print('Info gain',IG, info_gain)
                if IG > info_gain:
                    info_gain = IG
                    self.dim_split = idx
                    self.feature_uniq_split = value_at_branches.tolist()

                print('Current Values', self.dim_split,
                          self.feature_uniq_split)

            # elif IG == info_gain:
            #     print('equal')
            #     if len(self.feature_uniq_split) == len(value_at_branches):
            #         if self.dim_split > idx:
            #             self.dim_split = idx
            #             self.feature_uniq_split = value_at_branches.tolist()
            #     elif len(self.feature_uniq_split) < len(value_at_branches):
            #         self.dim_split = idx
            #         self.feature_uniq_split = value_at_branches.tolist()

        print('splitting at index', self.dim_split, value_at_branches,
              self.feature_uniq_split, IG, info_gain)
        xi = np.array(self.features)[:, self.dim_split]
        print('xi', xi)
        x = np.array(self.features, dtype=object)
        x[:, self.dim_split] = None
        # x = np.delete(self.features, self.dim_split, axis=1)
        print('x', x)

        for val in self.feature_uniq_split:
            indexes = np.where(xi == val)
            x_new = x[indexes].tolist()
            y_new = np.array(self.labels)[indexes].tolist()
            print("child - ", val, x_new, y_new)
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
            idx_child = self.feature_uniq_split.index(feature[self.dim_split])
            # feature = feature[:self.dim_split] + feature[self.dim_split + 1:]
            return self.children[idx_child].predict(feature)
        else:
            return self.cls_max