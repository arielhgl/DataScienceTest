from insuletchallenge import datasets
from insuletchallenge import models
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import concatenate
import numpy as np
import argparse
import pandas as pd

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-tr", "--train", type=str, required=True,
	help="path to train dataset")

ap.add_argument("-vl", "--pred", type=str, required=True,
	help="path to train dataset")

# load dataset attributes 
dt = datasets.load_train_attributes('training.csv')


# load images to train then normalize pixel values
img_train = datasets.load_images('training.csv')

img_train = img_train / 255.0



# split test and train data using 75% of data for training
split = train_test_split(dt, img_train, test_size=0.25)
(trainX, testX, trainImgX, testImgX) = split

# find largest value in prediction variable and then
# scale it in the range[0,1]
maxValue = dt['target'].max()
trainY = trainX['target']/maxValue
testY = testX['target']/maxValue

# process attributes to perform min-max scaling for continous
# features and stack them to categorical features
(trainX, testX) = datasets.process_train_attributes(trainX, testX)



# generate mlp and cnn architecture models
mlp = models.create_mlp(trainX.shape[1], regress=False)
cnn = models.create_cnn(512, 512, 3)#,regress=False)

# input of final layers as the output of both created 
# models (cnn and mlp)
combinedInput = concatenate([mlp.output, cnn.output])
# add fully connected layer with two dense 
# layers with the final one as out regressor
x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)

# final model accepts the two types of data for features extracted, also
# it accepts the images as input of the layer which at then end outputs a single value
# which is the variable that we are trying to predict
model = Model(inputs=[mlp.input, cnn.input], outputs=x)

# build the model architecture using mean_absolute_percentage_error
# as the loss function, which implies that we want to minimize the 
# percentage difference between our target predictions and the actual target value
opt = Adam()
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# train model
print("[INFO] training model...")
model.fit(x=[trainX, trainImgX], y=trainY,validation_data=([testX, testImgX], testY),epochs=20, batch_size=4)
# predictions
print("[INFO] predicting house prices...")
preds = model.predict([testX, testImgX])


# differences between predicted and actual from test data, 
# then compute % difference between predicten and actual target values
# and the absolute %diff
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)
# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)
# models stats
print("[INFO] avg. target value: {}, std target value: {}".format(
	dt["target"].mean(),
	dt["target"].std()))



print("[INFO] Absolute Percentage Difference mean: {:.2f}%, and std: {:.2f}%".format(mean, std))



# load input data to predict
dt_validate = datasets.load_predict_attributes('test.csv')

# load images then normalize pixel values for input prediction
img_train_validate = datasets.load_images('test.csv')
validateImgX = img_train_validate / 255.0

# process attributes and performs same transformations
# as with train and test data 
validateX = datasets.process_predict_attributes(dt_validate)

preds_predict = model.predict([validateX, validateImgX])


print(preds_predict.flatten()*maxValue)


gomez_answer = pd.read_csv('test.csv')

gomez_answer['target_predict'] = preds_predict.flatten()*maxValue

gomez_answer.to_csv('gomez-answer.csv')




data = pd.read_csv('training.csv')
images = datasets.load_images('training.csv')
data_predict = datasets.process_predict_attributes(data)
p = model.predict([data_predict, images])
data['predict_target'] = p.flatten()*maxValue

rmse = np.sqrt((np.sum(((np.array(data['target']) - np.array(data['predict_target']))**2)))/len(data['target']))
data.to_csv('p.csv')

print(rmse)

