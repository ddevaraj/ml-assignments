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
        # for test in features:
        #     predicted_labels.append(self.get_k_neighbors(test))
        # # print('labels', predicted_labels)
        # return predicted_labels
        # labels = []
        # for test in features:
        #     distances = []
        #     for train in self.features:
        #         distances.append(self.distance_function(train, test))
        #     indexes = np.argpartition(distances, self.k)
        #     votes = {}
        #     if self.k < 1:
        #         self.k = 1
        #     for i in range(self.k):
        #         label = self.labels[indexes[i]]
        #         if label not in votes.keys():
        #             votes[label] = 1
        #         else:
        #             votes[label] += 1
        #     labels.append(max(votes, key=votes.get))
        # return labels
        predicted_labels = []

        for x in features:
            actual_indices = self.get_k_neighbors(x)
            C0, C1 = 0, 0
            for j in actual_indices:
                if (self.labels[j] == 1):
                    C1 += 1
                else:
                    C0 += 1
            # assigning this feature to the class with highest number of votes
            if (C1 > C0):
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)
        return predicted_labels
           

    def get_k_neighbors(self, point: List[float]) -> List[int]:
        print('in getk')
        distances = []
        all_indices = []
        for i in range(len(self.features)):
            if (point != item for item in self.features[i]):
                dist = self.distance_function(point, self.features[i])
                distances.append(dist)
                all_indices.append(i)
        actual_indices = []
        dis_arr = np.asarray(distances)
        indices = dis_arr.argsort()[:self.k]

        for j in indices:
            actual_indices.append(all_indices[j])
        return actual_indices



if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
