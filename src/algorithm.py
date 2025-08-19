import numpy as np
from sympy.codegen.numpy_nodes import minimum


def euclidean_distance(known_point, new_point):
    difference = np.array(known_point) - np.array(new_point)
    squared_difference = difference ** 2
    sum = np.sum(squared_difference)
    distance = np.sqrt(sum)
    return distance

class kNN_classification:
    def __init__(self, train_set, test_set,val_set, k_set):
        self.train_set= train_set
        self.test_set = test_set
        self.val_set = val_set
        self.k_set = k_set

        # derived
        self.validation_point_data = []
        self.new_point_label_set = []


    def calculate_distances(self):
        # clear list of past data
        self.validation_point_data = []

        # compare new point to every known point
        for new_point_index in range(len(self.val_set)):
            distance_label_set = []
            for known_point in self.train_set:

                # get label
                label = known_point[-1]

                # get numeric features only
                known_point = known_point[:-1]
                new_point = self.val_set[new_point_index][:-1]

                # calculate distance
                distance = euclidean_distance(known_point,new_point)

                # append distance,label pair to a list
                distance_label_set.append([distance,label])

            '''
            this list contains 
            1. index of validation point
            2. list containing
                (distance to known point, label of known point) 
            '''
            self.validation_point_data.append([new_point_index,distance_label_set])


    def calculate_label(self):
        for new_point_index, distance_label_set in self.validation_point_data:
            # sorted in ascending order of distances
            distance_label_set_sorted = sorted(distance_label_set, key=lambda x: x[0], reverse=False)
            minimum_distance , predicted_label = distance_label_set_sorted[0]

            self.new_point_label_set.append([new_point_index,predicted_label])










