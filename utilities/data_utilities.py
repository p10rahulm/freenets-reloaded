from sklearn.model_selection import train_test_split
import json
import numpy as np


def split_data(x, y, test_size=0.1):
    return train_test_split(x, y, test_size=test_size, random_state=42)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)