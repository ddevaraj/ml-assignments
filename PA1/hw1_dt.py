import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
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
        def parent_entropy(labels):
            n = len(labels)
            unique_labels = np.unique(labels)
            sums = 0
            for label in unique_labels:
                no_instances = labels.count(label)
                p = float(no_instances) / float(n)
                sums += -(p * np.log2(p))
            return sums

        S = parent_entropy(self.labels)
        for idx in range(len(self.features[0])):
            if "max_infogain" not in locals():
                max_infogain = float('-inf')
            values_at_dimensions = np.array(self.features)[:, idx]
            if None not in values_at_dimensions:
                value_at_branches = np.unique(values_at_dimensions)
                branches = np.zeros((self.num_cls, len(value_at_branches)))
                attribute_mapping = dict() #map branches to index
                j = 0
                for attribute in value_at_branches:
                    attribute_mapping[attribute] = j
                    j += 1

                i = 0
                label_mapping = dict()
                labels = list(set(self.labels))
                for l in labels:
                    label_mapping[l] = i
                    i += 1

                for index in range(len(self.features)):
                    col = attribute_mapping[self.features[index][idx]]
                    row = label_mapping[self.labels[index]]
                    branches[row][col] += 1
                IG = Util.Information_Gain(S,branches.T)
                if IG > max_infogain:
                    max_infogain = IG
                    self.dim_split = idx
                    self.feature_uniq_split = value_at_branches.tolist()
                elif IG == max_infogain and self.dim_split != idx:
                    if len(value_at_branches) > len(self.feature_uniq_split):
                        max_infogain = IG
                        self.dim_split = idx
                        self.feature_uniq_split = value_at_branches.tolist()
                    elif len(value_at_branches) == len(self.feature_uniq_split):
                        if idx < self.dim_split:
                            self.dim_split = idx
                            max_infogain = IG
                            self.feature_uniq_split = value_at_branches.tolist()
        x_at_i = np.array(self.features)[:, self.dim_split]
        arr = np.array(self.features, dtype=object)
        arr[:, self.dim_split] = None

        for val in self.feature_uniq_split:
            indexes = np.where(x_at_i == val)
            x_new = arr[indexes].tolist()
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
            id_child = self.feature_uniq_split.index(feature[self.dim_split])
            return self.children[id_child].predict(feature)
        else:
            return self.cls_max