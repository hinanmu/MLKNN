#@Time      :2018/9/14 13:39
#@Author    :zhounan
# @FileName: data_process.py

import pandas as pd
import numpy as np
from scipy.io import arff

train_file_path = 'yeast_corpus/yeast-train.arff'
test_file_path = 'yeast_corpus/yeast-test.arff'

train_data, meta = arff.loadarff(train_file_path)
test_data, meta = arff.loadarff(test_file_path)

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

train_data = train_df.values
test_data = test_df.values

train_x = np.array(train_data[:, 0:103])
train_y = np.array(train_data[:, 104:117])
test_x = np.array(test_data[:, 0:103])
test_y = np.array(test_data[:, 104:117])

np.save('dataset/train_x.npy', train_x)
np.save('dataset/train_y.npy', train_y)
np.save('dataset/test_x.npy', test_x)
np.save('dataset/test_y.npy', test_y)
