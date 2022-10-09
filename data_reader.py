import numpy as np


def read_data():
    row_data = []
    with open('data.dat') as file:
        for line in file:
            row_data.append(line.split(None))
        row_data.remove([])
    row_data = np.array(row_data, dtype=str)
    labels = row_data[1:, 1]
    data = row_data[1:, 2:]
    data = np.array(data, dtype=int)
    return labels, data


def get_row_data():
    row_data = []
    with open('data.dat') as file:
        for line in file:
            row_data.append(line.split(None))
        row_data.remove([])
    row_data = np.array(row_data, dtype=str)
    row_data = row_data[1:, :]
    return row_data
