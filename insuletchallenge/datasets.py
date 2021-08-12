# import the necessary packages
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os
import datetime as dt



def load_train_attributes(inputPathTrain):
	# determine columns to be loaded when reading the file
    #(columns were selected based on the feature engeneering analysis
    #on the "feature_eng" file)
    # and then load it using pandas
    cols = ['date','bar','baz', 'xgt', 'qgg', 'lux', 'wsg', 'yyz', 'drt', 'gox', 'foo',
       'boz', 'fyt', 'lgh', 'hrt', 'juu','target']
    df = pd.read_csv(inputPathTrain, usecols = cols)

    return df

def load_predict_attributes(inputPathTrain):
	# determine columns to be loaded when reading the file
    #(columns were selected based on the feature engeneering analysis
    #on the "feature_eng" file)
    # and then load it using pandas
    cols = ['date','bar','baz','xgt', 'qgg', 'lux', 'wsg', 'yyz', 'drt', 'gox', 'foo',
       'boz', 'fyt', 'lgh', 'hrt', 'juu']
    df = pd.read_csv(inputPathTrain, usecols = cols)

    return df

def load_images(inputPath):

    df = pd.read_csv(inputPath, usecols=[0])

    imagesPaths = list(df.iloc[:,0])

    inputImages = []

    for imagePath in imagesPaths:
        # load the input image, resize it to be 512 by 512, and then
        # update the list of input images
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)
        inputImages.append(image)


    return np.array(inputImages)


def process_train_attributes(train, test):
	# initialize the column names of the continuous data

	# PCA analysis variables
	continuous = ['bar', 'xgt', 'qgg', 'lux', 'wsg', 'yyz', 'drt']

	# correlation analysis variables
	# continuous = ['xgt','bar', 'hrt', 'boz', 'lux','qgg','yyz']
	# continuous = ['xgt','bar', 'hrt']
	
	# performin min-max scaling each continuous feature column to
	# the range [0, 1]
	cs = MinMaxScaler()
	trainContinuous = cs.fit_transform(train[continuous])
	testContinuous = cs.transform(test[continuous])
	# one-hot encode the zip code categorical data (by definition of
	# one-hot encoding, all output features are now in the range [0, 1])
	categorical = ['baz','fyt','lgh']

	trainCategorical = train[categorical]
	testCategorical = test[categorical]
	
	
	trainDate = pd.DataFrame()
	trainDate['day'] = pd.DataFrame(pd.to_datetime(train['date'], format='%Y-%m-%d').dt.weekday)
	trainDate['month'] = pd.DataFrame(pd.to_datetime(train['date'], format='%Y-%m-%d').dt.month)

	trainDate = cs.fit_transform(trainDate)

	testDate = pd.DataFrame()
	testDate['day'] = pd.DataFrame(pd.to_datetime(test['date'], format='%Y-%m-%d').dt.weekday)
	testDate['month'] = pd.DataFrame(pd.to_datetime(test['date'], format='%Y-%m-%d').dt.month)

	testDate = cs.fit_transform(testDate)

	# construct our training and testing data points by concatenating
	# the categorical features with the continuous features
	trainX = np.hstack([trainCategorical, trainContinuous,trainDate])
	testX = np.hstack([testCategorical, testContinuous, testDate])
	# return the concatenated training and testing data
	return (trainX, testX)

    
def process_predict_attributes(valdt):
	# initialize the column names of the continuous data

	# PCA analysis variables
	continuous = ['bar', 'xgt', 'qgg', 'lux', 'wsg', 'yyz', 'drt']

	# correlation analysis variables
	# continuous = ['xgt','bar', 'hrt', 'boz', 'lux','qgg','yyz']
	# continuous = ['xgt','bar', 'hrt']

	# performin min-max scaling each continuous feature column to
	# the range [0, 1]
	cs = MinMaxScaler()
	valContinuous = cs.fit_transform(valdt[continuous])
	# one-hot encode the zip code categorical data (by definition of
	# one-hot encoding, all output features are now in the range [0, 1])
	categorical = ['baz','fyt','lgh']

	valCategorical = valdt[categorical]

	valDate = pd.DataFrame()
	valDate['day'] = pd.DataFrame(pd.to_datetime(valdt['date'], format='%Y-%m-%d').dt.weekday)
	valDate['month'] = pd.DataFrame(pd.to_datetime(valdt['date'], format='%Y-%m-%d').dt.month)

	valDate = cs.fit_transform(valDate)

	# construct our training and testing data points by concatenating
	# the categorical features with the continuous features
	valX = np.hstack([valContinuous, valCategorical, valDate])
	# return the concatenated training and testing data
	return valX
