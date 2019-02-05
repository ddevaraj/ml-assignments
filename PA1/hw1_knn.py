from __future__ import division, print_function

from typing import List

import numpy as np
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:
    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    def train(self, features: List[List[float]], labels: List[int]):
        # features: List[List[float]] a list of points
        # labels: List[int] labels of features
        self.features = features
        self.labels = labels

    def predict(self, features: List[List[float]]) -> List[int]:
        # features: List[List[float]] a list of points
        # return: List[int] a list of predicted labels
        # predicted_labels = []

        predicted_labels = []

        for x in features:
            actual_indices = self.get_k_neighbors(x)
            # to classify the features into 2 classes {0,1}
            class0, class1 = 0, 0
            for j in actual_indices:
                if (self.labels[j] == 1):
                    class1 += 1
                else:
                    class0 += 1
            # assigning this feature to the class with highest number of votes
            if (class1 > class0):
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)
        return predicted_labels

    def get_k_neighbors(self, point: List[float]) -> List[int]:
        distances = []
        index_arr = []
        for i in range(len(self.features)):
            if (point != item for item in self.features[i]):
                dist = self.distance_function(point, self.features[i])
                distances.append(dist)
                index_arr.append(i)
        k_neighbor_label = []
        dis_arr = np.asarray(distances)
        indices = dis_arr.argsort()[:self.k]

        for j in indices:
            k_neighbor_label.append(index_arr[j])
        return k_neighbor_label


if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
