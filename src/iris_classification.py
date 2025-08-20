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
    class_set = list(set(np.asarray(data)[:,-1].tolist()))
    train_set, test_set, val_set = partition(data=data, train_percentage=80,val_percentage=20,test_percentage=0)

    # kNN
    model = kNN_classification(
        train_set,
        test_set,
        val_set,
        class_set=class_set,
        k_set=np.arange(1, 45, 2)
    )

    model.calculate_distances()
    model.calculate_k_stats()

    for k_result in model.kNN_Result_set:
        k_result.summarize()