from utils import parse_value
import numpy as np
from algorithm import kNN_classification
from utils import load_data, partition
DATA_INPUT_PATH = "../data/iris/iris.data"

# data structure
'''
Data Format (left most column to right most column):
    1. sepal length in cm
    2. sepal width in cm 
    3. petal length in cm 
    4. petal width in cm
    5. class (Iris Setosa, Iris Versicolour, Iris Virginica)
'''

if __name__ == '__main__':
    # loading data
    file = open(DATA_INPUT_PATH, 'r')
    data = []
    for line in file:
        # removes whitespaces
        line = line.strip()

        # skip rest of loop and move to next line, if line empty
        if not line:
            continue

        # splits by comma
        parts = line.split(",")  # split by comma

        row = []
        for part in parts:
            parsed_value = parse_value(part)
            row.append(parsed_value)
        data.append(row)

    # shuffle data
    np.random.seed(42)  # set seed for reproducibility
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

    model.calculate_distances()
    model.calculate_label()
    print(model.new_point_label_set)
