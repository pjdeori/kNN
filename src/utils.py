import numpy as np

def parse_value(value):
    # check if integer
    try:
        return int(value)
    except ValueError:
        pass
    # check if float
    try:
        return float(value)
    except ValueError:
        pass
    # return as string if not numeric
    return value


def load_data(path):
    file = open(path,'r')
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
    return data

def partition(data, train_percentage=70,test_percentage=15,val_percentage=15):
    number_of_rows = len(data)

    if train_percentage + test_percentage + val_percentage == 100:
        train_index = int(number_of_rows * (train_percentage / 100))  # index at 70% of data
        test_index = int(number_of_rows * ((train_percentage + test_percentage) / 100))  # index at (80+15) 95% of data
        val_index = int(number_of_rows * ((
                                                  train_percentage + test_percentage + val_percentage) / 100))  # index at (80+15+15)% or 100% of data

        train_set = data[:train_index]
        test_set = data[train_index:test_index]
        val_set = data[test_index:val_index]
        return train_set,test_set,val_set
    else:
        print('partitioning failed, error in splitting ratios')

def get_majority_class(k_labels):
    freq = {}
    for label in k_labels:
        if label not in freq:
            freq[label] = 0
        freq[label] += 1

    # get majority class (highest frequency)
    majority_class = max(freq, key=freq.get)
    return majority_class



