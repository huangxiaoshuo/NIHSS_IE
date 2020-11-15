import sys
sys.path.append('..')
from main.common.tools import load_pickle
class Data(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.read_data(self.data_path)

    def read_data(self, path):
        if 'pkl' in str(path):
            data = load_pickle(path)
        return data
