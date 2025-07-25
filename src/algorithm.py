import numpy as np

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

        self.validation_point_data = []


    def calculate_distances_of_new_points_to_every_known_points(self):
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
                (label of known point , distance to known point) 
            '''
            self.validation_point_data.append([new_point_index,distance_label_set])


    # def label_new_points_for_every_k_samples(self):







