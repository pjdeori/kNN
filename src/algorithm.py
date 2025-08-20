import numpy as np
from typing import List
from src.utils import get_majority_class

def euclidean_distance(known_point, new_point):
    difference = np.array(known_point) - np.array(new_point)
    squared_difference = difference ** 2
    sum = np.sum(squared_difference)
    distance = np.sqrt(sum)
    return distance

class kNN_classification:
    def __init__(self, train_set, test_set,val_set, k_set, class_set):
        self.train_set= train_set
        self.test_set = test_set
        self.val_set = val_set
        self.k_set = k_set
        self.class_set = class_set

        # derived
        self.validation_point_data = [] # [[index, [[distance, label],...]],...]
        self.k_stats =[] # [k, accuracy]
        self.kNN_Result_set: List[kNN_Result]=[]

    def calculate_distances(self):
        # clear list of past data
        self.validation_point_data = []

        # compare new point to every known point
        for new_point_index in range(len(self.val_set)):
            distance_label_set= []
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

            # sort distances in ascending order
            distance_label_set = sorted(distance_label_set, key=lambda x: x[0], reverse=False)
            sorted_label_set = np.asarray(distance_label_set)[:,-1].tolist()

            self.validation_point_data.append([new_point_index,sorted_label_set])

    def calculate_k_stats(self):
        for k in self.k_set:
            k_result = kNN_Result(k,self.class_set) # to save results
            for new_point_index,sorted_labels in self.validation_point_data:
                # get k samples
                k_samples = sorted_labels[0:k]

                # get majority class of k samples
                majority_class = get_majority_class(k_samples)

                # save the result
                k_result.result_datapoint(self.val_set[new_point_index][-1],majority_class)
            self.kNN_Result_set.append(k_result)

class Label_Stat:
    def __init__(self, label):
        self.label = label
        self.true_pos_count = 0
        self.true_neg_count = 0
        self.false_pos_count = 0
        self.false_neg_count = 0

class kNN_Result:
    def __init__(self,k, class_set):
        self.k = k
        self.class_set = class_set

        # derived
        self.class_result_set: List[Label_Stat] = []
        for label in class_set:
            self.class_result_set.append(Label_Stat(label))

    def result_datapoint(self, truth, prediction):
        for class_result in self.class_result_set:
            if class_result.label == truth and truth == prediction:
                class_result.true_pos_count += 1
            elif class_result.label == truth and truth != prediction:
                class_result.false_neg_count += 1
            elif class_result.label == prediction and prediction != truth:
                class_result.false_pos_count += 1
            else:
                class_result.true_neg_count += 1

    def summarize(self):
        total_tp = sum(c.true_pos_count for c in self.class_result_set)
        total_tn = sum(c.true_neg_count for c in self.class_result_set)
        total_fp = sum(c.false_pos_count for c in self.class_result_set)
        total_fn = sum(c.false_neg_count for c in self.class_result_set)
        total_samples = total_tp + total_tn + total_fp + total_fn
        accuracy = total_tp / total_samples if total_samples > 0 else 0

        print(
            f"k: {self.k}, Accuracy: {accuracy:.4f}, TP: {total_tp}, TN: {total_tn}, FP: {total_fp}, FN: {total_fn}, Total samples: {total_samples}")








