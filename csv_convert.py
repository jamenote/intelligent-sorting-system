import numpy as np
import pandas as pd


def csv_to_Matrix(path):
    np.set_printoptions(suppress=True)
    pd.set_option('display.float_format', lambda x: '%.8f'% x)
    x_Matrix = pd.read_csv(path, header=None, dtype=float)
    x_Matrix = np.array(x_Matrix)
    return x_Matrix

