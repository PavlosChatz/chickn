import numpy as np

from Models import *
from Data import load_data_animals10


if __name__ == '__main__':
    dim = (64, 64)
    X_train, Y_train, X_val, Y_val = load_data_animals10(dim)
    model_helper(X_train, Y_train, X_val, Y_val)
