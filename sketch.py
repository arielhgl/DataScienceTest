import cv2
import pandas as pd
import numpy as np


inputPath = 'training.csv'
paths = list(pd.read_csv(inputPath, usecols=[0]).iloc[:,0])


list_dims = []
for i in paths:
    dims = cv2.imread(i).shape
    list_dims.append(dims)



np.mean([x[1] for x in list_dims])

np.mean([x[0] for x in list_dims])
