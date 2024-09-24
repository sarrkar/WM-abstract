import numpy as np

def get_one_hot(index: int, total: int = 18):
    result = np.zeros(total)
    result[index-1] = 1
    return result