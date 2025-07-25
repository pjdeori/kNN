import numpy as np

from algorithm import kNN_classification
from utils import load_data, partition

if __name__ == '__main__':
    '''
    Data Format (left most column to right most column):
        1. sepal length in cm
        2. sepal width in cm 
        3. petal length in cm 
        4. petal width in cm
        5. class (Iris Setosa, Iris Versicolour, Iris Virginica)
    '''

    # returns list of data
    data = load_data('../data/iris/iris.data')

    # shuffle data
    np.random.seed(42) # set seed for reproducibility
    np.random.shuffle(data)

    # partitioning
    train_set, test_set, val_set = partition(data=data)

    # kNN
    model = kNN_classification(
        train_set,
        test_set,
        val_set,
        k_set=[1, 3, 5, 10, 15]
    )

    '''
    the class methods are intentionally structured to resemble pseudo code algorithm
    '''
    model.calculate_distances_of_new_points_to_every_known_points()
    # model.label_new_points_for_every_k_samples()




