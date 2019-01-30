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
        # print('in predict')
        predicted_labels = []

        for x in features:
            actual_indices = self.get_k_neighbors(x)
            class1, class2 = 0, 0
            for j in actual_indices:
                if (self.labels[j] == 1):
                    class1 += 1
                else:
                    class2 += 1
            if (class1 > class2):
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)
        # print(predicted_labels)
        return predicted_labels
           

    def get_k_neighbors(self, point: List[float]) -> List[int]:
        # print('in getkneighbors')
        distances = []
        point_index = []
        for i in range(len(self.features)):
            if (point != item for item in self.features[i]):
                distance = self.distance_function(point, self.features[i])
                distances.append(distance)
                point_index.append(i)
        k_neighbor_index = []
        dis_arr = np.asarray(distances)
        sorted_dist = dis_arr.argsort()[:self.k]

        for j in sorted_dist:
            k_neighbor_index.append(point_index[j])
        # print(k_neighbor_index)
        return k_neighbor_index



if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
